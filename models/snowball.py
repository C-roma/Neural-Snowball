import sys
import numpy as np
import random
sys.path.append('..')
import nrekit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import sklearn.metrics 
import copy

class DynamicThreshold:
    def __init__(self, alpha=0.5, beta=3, decay_rate=0.9):
        """
        Initialize dynamic threshold parameters.
        Args:
            alpha (float): Initial confidence threshold.
            beta (int): Initial number of iterations for instance selection.
            decay_rate (float): Decay factor for thresholds.
        """
        self.alpha = alpha
        self.beta = beta
        self.decay_rate = decay_rate
        self.confidence_history = []

    def adjust_thresholds(self, avg_confidence, low_threshold=0.4, high_threshold=0.7, step=1):
        """
        Adjust alpha and beta based on average confidence.
        """
        if avg_confidence < low_threshold:
            self.beta += step  # Increase exploration
        elif avg_confidence > high_threshold:
            self.beta = max(self.beta - step, 1)  # Reduce unnecessary iterations
        self.alpha = avg_confidence  # Update alpha to match current confidence

    def decay_thresholds(self):
        """
        Apply decay to thresholds.
        """
        self.alpha *= self.decay_rate
        self.beta = max(1, int(self.beta * self.decay_rate))

    def check_early_stopping(self, window_size=5, variance_threshold=0.001):
        """
        Check if confidence scores have stabilized for early stopping.
        """
        if len(self.confidence_history) < window_size:
            return False
        recent_scores = self.confidence_history[-window_size:]
        return np.var(recent_scores) < variance_threshold
    
class Siamese(nn.Module):

    def __init__(self, sentence_encoder, hidden_size=230, drop_rate=0.5, pre_rep=None, euc=True):
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder  # Should be different from the main sentence encoder
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, 1)
        self.cost = nn.BCELoss(reduction="none")
        self.drop = nn.Dropout(drop_rate)
        self._accuracy = 0.0
        self.pre_rep = pre_rep
        self.euc = euc  # Use Euclidean if True, otherwise use Cosine Similarity

    def forward(self, data, num_size, num_class, threshold=0.5):
        x = self.sentence_encoder(data).contiguous().view(num_class, num_size, -1)
        x1 = x[:, :num_size // 2].contiguous().view(-1, self.hidden_size)
        x2 = x[:, num_size // 2:].contiguous().view(-1, self.hidden_size)
        y1 = x[:num_class // 2, :].contiguous().view(-1, self.hidden_size)
        y2 = x[num_class // 2:, :].contiguous().view(-1, self.hidden_size)

        label = torch.zeros((x1.size(0) + y1.size(0))).long().cuda()
        label[:x1.size(0)] = 1
        z1 = torch.cat([x1, y1], 0)
        z2 = torch.cat([x2, y2], 0)

        if self.euc:
            dis = torch.pow(z1 - z2, 2)
        else:
            dis = F.cosine_similarity(z1, z2, dim=1).unsqueeze(-1)
            dis = 1 - dis  # Convert similarity to a distance measure

        dis = self.drop(dis)
        score = torch.sigmoid(self.fc(dis).squeeze())

        self._loss = self.cost(score, label.float()).mean()
        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        self._accuracy = torch.mean((pred == label).type(torch.FloatTensor))
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        self._prec = float(np.logical_and(pred == 1, label == 1).sum()) / float((pred == 1).sum() + 1)
        self._recall = float(np.logical_and(pred == 1, label == 1).sum()) / float((label == 1).sum() + 1)

    def encode(self, dataset, batch_size=0):
        self.sentence_encoder.eval()
        with torch.no_grad():
            if self.pre_rep is not None:
                return self.pre_rep[dataset['id'].view(-1)]

            if batch_size == 0:
                x = self.sentence_encoder(dataset)
            else:
                total_length = dataset['word'].size(0)
                max_iter = total_length // batch_size
                if total_length % batch_size != 0:
                    max_iter += 1
                x = []
                for it in range(max_iter):
                    scope = list(range(batch_size * it, min(batch_size * (it + 1), total_length)))
                    with torch.no_grad():
                        _ = {'word': dataset['word'][scope], 'mask': dataset['mask'][scope]}
                        if 'pos1' in dataset:
                            _['pos1'] = dataset['pos1'][scope]
                            _['pos2'] = dataset['pos2'][scope]
                        _x = self.sentence_encoder(_)
                    x.append(_x.detach())
                x = torch.cat(x, 0)
            return x

    def forward_infer(self, x, y, threshold=0.5, batch_size=0):
        x = self.encode(x, batch_size=batch_size)
        support_size = x.size(0)
        y = self.encode(y, batch_size=batch_size)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

        if self.euc:
            dis = torch.pow(x - y, 2)
        else:
            dis = F.cosine_similarity(x, y, dim=-1).unsqueeze(-1)
            dis = 1 - dis

        score = torch.sigmoid(self.fc(dis).squeeze(-1)).mean(0)

        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        pred = pred.view(support_size, -1).sum(0)
        pred[pred < 1] = 0
        pred[pred > 0] = 1
        return pred

    def forward_infer_sort(self, x, y, batch_size=0):
        x = self.encode(x, batch_size=batch_size)
        support_size = x.size(0)
        y = self.encode(y, batch_size=batch_size)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

        if self.euc:
            dis = torch.pow(x - y, 2)
        else:
            dis = F.cosine_similarity(x, y, dim=-1).unsqueeze(-1)
            dis = 1 - dis

        score = torch.sigmoid(self.fc(dis).squeeze(-1)).mean(0)

        pred = []
        for i in range(score.size(0)):
            pred.append((score[i], i))
        pred.sort(key=lambda x: x[0], reverse=True)
        return pred


class Snowball(nrekit.framework.Model):
    
    def __init__(self, sentence_encoder, base_class, siamese_model, hidden_size=230, drop_rate=0.5, weight_table=None, pre_rep=None, neg_loader=None, args=None):
        nrekit.framework.Model.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.base_class = base_class
        self.fc = nn.Linear(hidden_size, base_class)
        self.drop = nn.Dropout(drop_rate)
        self.siamese_model = siamese_model
        # self.cost = nn.BCEWithLogitsLoss()
        self.cost = nn.BCELoss(reduction="none")
        # self.cost = nn.CrossEntropyLoss()
        self.weight_table = weight_table
        
        self.args = args

        self.pre_rep = pre_rep
        self.neg_loader = neg_loader
        self.dynamic_threshold = DynamicThreshold()  # Instantiate DynamicThreshold


    # def __loss__(self, logits, label):
    #     onehot_label = torch.zeros(logits.size()).cuda()
    #     onehot_label.scatter_(1, label.view(-1, 1), 1)
    #     return self.cost(logits, onehot_label)

    # def __loss__(self, logits, label):
    #     return self.cost(logits, label)

    def forward_base(self, data):
        batch_size = data['word'].size(0)
        x = self.sentence_encoder(data) # (batch_size, hidden_size)
        x = self.drop(x)
        x = self.fc(x) # (batch_size, base_class)

        x = torch.sigmoid(x)
        if self.weight_table is None:
            weight = 1.0
        else:
            weight = self.weight_table[data['label']].unsqueeze(1).expand(-1, self.base_class).contiguous().view(-1)
        label = torch.zeros((batch_size, self.base_class)).cuda()
        label.scatter_(1, data['label'].view(-1, 1), 1) # (batch_size, base_class)
        loss_array = self.__loss__(x, label)
        self._loss = ((label.view(-1) + 1.0 / self.base_class) * weight * loss_array).mean() * self.base_class
        # self._loss = self.__loss__(x, data['label'])
        
        _, pred = x.max(-1)
        self._accuracy = self.__accuracy__(pred, data['label'])
        self._pred = pred
    
    def forward_baseline(self, support_pos, query, threshold=0.5):
        '''
        baseline model
        support_pos: positive support set
        support_neg: negative support set
        query: query set
        threshold: ins whose prob > threshold are predicted as positive
        '''
        
        # train
        self._train_finetune_init()
        # support_rep = self.encode(support, self.args.infer_batch_size)
        support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)
        # self._train_finetune(support_rep, support['label'])
        self._train_finetune(support_pos_rep)

        
        # test
        query_prob = self._infer(query, batch_size=self.args.infer_batch_size).cpu().detach().numpy()
        label = query['label'].cpu().detach().numpy()
        self._baseline_accuracy = float(np.logical_or(np.logical_and(query_prob > threshold, label == 1), np.logical_and(query_prob < threshold, label == 0)).sum()) / float(query_prob.shape[0])
        if (query_prob > threshold).sum() == 0:
            self._baseline_prec = 0
        else:        
            self._baseline_prec = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((query_prob > threshold).sum())
        self._baseline_recall = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((label == 1).sum())
        if self._baseline_prec + self._baseline_recall == 0:
            self._baseline_f1 = 0
        else:
            self._baseline_f1 = float(2.0 * self._baseline_prec * self._baseline_recall) / float(self._baseline_prec + self._baseline_recall)
        self._baseline_auc = sklearn.metrics.roc_auc_score(label, query_prob)
        if self.args.print_debug:
            print('')
            sys.stdout.write('[BASELINE EVAL] acc: {0:2.2f}%, prec: {1:2.2f}%, rec: {2:2.2f}%, f1: {3:1.3f}, auc: {4:1.3f}'.format( \
                self._baseline_accuracy * 100, self._baseline_prec * 100, self._baseline_recall * 100, self._baseline_f1, self._baseline_auc))
            print('')

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward_few_shot_baseline(self, support, query, label, B, N, K, Q):
        support_rep = self.encode(support, self.args.infer_batch_size)
        query_rep = self.encode(query, self.args.infer_batch_size)
        support_rep.view(B, N, K, -1)
        query_rep.view(B, N * Q, -1)
        
        NQ = N * Q
         
        # Prototypical Networks 
        proto = torch.mean(support_rep, 2) # Calculate prototype for each class
        logits = -self.__batch_dist__(proto, query)
        _, pred = torch.max(logits.view(-1, N), 1)

        self._accuracy = self.__accuracy__(pred.view(-1), label.view(-1))

        return logits, pred

#    def forward_few_shot(self, support, query, label, B, N, K, Q):
#        for b in range(B):
#            for n in range(N):
#                _forward_train(self, support_pos, None, query, distant, threshold=0.5):
#
#        '''
#        support_rep = self.encode(support, self.args.infer_batch_size)
#        query_rep = self.encode(query, self.args.infer_batch_size)
#        support_rep.view(B, N, K, -1)
#        query_rep.view(B, N * Q, -1)
#        '''
#        
#        proto = []
#        for b in range(B):
#            for N in range(N)
#        
#        NQ = N * Q
#         
#        # Prototypical Networks 
#        proto = torch.mean(support_rep, 2) # Calculate prototype for each class
#        logits = -self.__batch_dist__(proto, query)
#        _, pred = torch.max(logits.view(-1, N), 1)
#
#        self._accuracy = self.__accuracy__(pred.view(-1), label.view(-1))
#
#        return logits, pred

    def _train_finetune_init(self):
        # init variables and optimizer
        self.new_W = Variable(self.fc.weight.mean(0) / 1e3, requires_grad=True)
        self.new_bias = Variable(torch.zeros((1)), requires_grad=True)
        self.optimizer = optim.Adam([self.new_W, self.new_bias], self.args.finetune_lr, weight_decay=self.args.finetune_wd)
        self.new_W = self.new_W.cuda()
        self.new_bias = self.new_bias.cuda()

    def _train_finetune(self, data_repre, learning_rate=None, weight_decay=1e-5):
        '''
        train finetune classifier with given data
        data_repre: sentence representation (encoder's output)
        label: label
        '''
        
        self.train()

        optimizer = self.optimizer
        if learning_rate is not None:
            optimizer = optim.Adam([self.new_W, self.new_bias], learning_rate, weight_decay=weight_decay)

        # hyperparameters
        max_epoch = self.args.finetune_epoch
        batch_size = self.args.finetune_batch_size
        
        # dropout
        data_repre = self.drop(data_repre) 
        
        # train
        if self.args.print_debug:
            print('')
        for epoch in range(max_epoch):
            max_iter = data_repre.size(0) // batch_size
            if data_repre.size(0) % batch_size != 0:
                max_iter += 1
            order = list(range(data_repre.size(0)))
            random.shuffle(order)
            for i in range(max_iter):            
                x = data_repre[order[i * batch_size : min((i + 1) * batch_size, data_repre.size(0))]]
                # batch_label = label[order[i * batch_size : min((i + 1) * batch_size, data_repre.size(0))]]
                
                # neg sampling
                # ---------------------
                batch_label = torch.ones((x.size(0))).long().cuda()
                neg_size = int(x.size(0) * 1)
                neg = self.neg_loader.next_batch(neg_size)
                neg = self.encode(neg, self.args.infer_batch_size)
                x = torch.cat([x, neg], 0)
                batch_label = torch.cat([batch_label, torch.zeros((neg_size)).long().cuda()], 0)
                # ---------------------

                x = torch.matmul(x, self.new_W) + self.new_bias # (batch_size, 1)
                x = torch.sigmoid(x)

                # iter_loss = self.__loss__(x, batch_label.float()).mean()
                weight = torch.ones(batch_label.size(0)).float().cuda()
                weight[batch_label == 0] = self.args.finetune_weight #1 / float(max_epoch)
                iter_loss = (self.__loss__(x, batch_label.float()) * weight).mean()

                optimizer.zero_grad()
                iter_loss.backward(retain_graph=True)
                optimizer.step()
                if self.args.print_debug:
                    sys.stdout.write('[snowball finetune] epoch {0:4} iter {1:4} | loss: {2:2.6f}'.format(epoch, i, iter_loss) + '\r')
                    sys.stdout.flush()
        self.eval()

    def _add_ins_to_data(self, dataset_dst, dataset_src, ins_id, label=None):
        '''
        add one instance from dataset_src to dataset_dst (list)
        dataset_dst: destination dataset
        dataset_src: source dataset
        ins_id: id of the instance
        '''
        dataset_dst['word'].append(dataset_src['word'][ins_id])
        if 'pos1' in dataset_src:
            dataset_dst['pos1'].append(dataset_src['pos1'][ins_id])
            dataset_dst['pos2'].append(dataset_src['pos2'][ins_id])
        dataset_dst['mask'].append(dataset_src['mask'][ins_id])
        if 'id' in dataset_dst and 'id' in dataset_src:
            dataset_dst['id'].append(dataset_src['id'][ins_id])
        if 'entpair' in dataset_dst and 'entpair' in dataset_src:
            dataset_dst['entpair'].append(dataset_src['entpair'][ins_id])
        if 'label' in dataset_dst and label is not None:
            dataset_dst['label'].append(label)

    def _add_ins_to_vdata(self, dataset_dst, dataset_src, ins_id, label=None):
        '''
        add one instance from dataset_src to dataset_dst (variable)
        dataset_dst: destination dataset
        dataset_src: source dataset
        ins_id: id of the instance
        '''
        dataset_dst['word'] = torch.cat([dataset_dst['word'], dataset_src['word'][ins_id].unsqueeze(0)], 0)
        if 'pos1' in dataset_src:
            dataset_dst['pos1'] = torch.cat([dataset_dst['pos1'], dataset_src['pos1'][ins_id].unsqueeze(0)], 0)
            dataset_dst['pos2'] = torch.cat([dataset_dst['pos2'], dataset_src['pos2'][ins_id].unsqueeze(0)], 0)
        dataset_dst['mask'] = torch.cat([dataset_dst['mask'], dataset_src['mask'][ins_id].unsqueeze(0)], 0)
        if 'id' in dataset_dst and 'id' in dataset_src:
            dataset_dst['id'] = torch.cat([dataset_dst['id'], dataset_src['id'][ins_id].unsqueeze(0)], 0)
        if 'entpair' in dataset_dst and 'entpair' in dataset_src:
            dataset_dst['entpair'].append(dataset_src['entpair'][ins_id])
        if 'label' in dataset_dst and label is not None:
            dataset_dst['label'] = torch.cat([dataset_dst['label'], torch.ones((1)).long().cuda()], 0)

    def _dataset_stack_and_cuda(self, dataset):
        '''
        stack the dataset to torch.Tensor and use cuda mode
        dataset: target dataset
        '''
        if (len(dataset['word']) == 0):
            return
        dataset['word'] = torch.stack(dataset['word'], 0).cuda()
        if 'pos1' in dataset:
            dataset['pos1'] = torch.stack(dataset['pos1'], 0).cuda()
            dataset['pos2'] = torch.stack(dataset['pos2'], 0).cuda()
        dataset['mask'] = torch.stack(dataset['mask'], 0).cuda()
        dataset['id'] = torch.stack(dataset['id'], 0).cuda()

    def encode(self, dataset, batch_size=0):
        self.sentence_encoder.eval()
        with torch.no_grad():
            if self.pre_rep is not None:
                return self.pre_rep[dataset['id'].view(-1)]

            if batch_size == 0:
                x = self.sentence_encoder(dataset)
            else:
                total_length = dataset['word'].size(0)
                max_iter = total_length // batch_size
                if total_length % batch_size != 0:
                    max_iter += 1
                x = []
                for it in range(max_iter):
                    scope = list(range(batch_size * it, min(batch_size * (it + 1), total_length)))
                    with torch.no_grad():
                        _ = {'word': dataset['word'][scope], 'mask': dataset['mask'][scope]}
                        if 'pos1' in dataset:
                            _['pos1'] = dataset['pos1'][scope]
                            _['pos2'] = dataset['pos2'][scope]
                        _x = self.sentence_encoder(_)
                    x.append(_x.detach())
                x = torch.cat(x, 0)
            return x

    def _infer(self, dataset, batch_size=0):
        '''
        get prob output of the finetune network with the input dataset
        dataset: input dataset
        return: prob output of the finetune network
        '''
        x = self.encode(dataset, batch_size=batch_size) 
        x = torch.matmul(x, self.new_W) + self.new_bias # (batch_size, 1)
        x = torch.sigmoid(x)
        return x.view(-1)

    def _forward_train(self, support_pos, query, distant, threshold=0.5):
    '''
    Enhanced Snowball process with top-k approach
    '''
    # Existing initialization code remains the same
    snowball_max_iter = self.args.snowball_max_iter
    candidate_num_class = 20
    candidate_num_ins_per_class = 100
    
    # Top-k related parameters
    top_k = self.args.top_k  # New parameter to control the number of top similar instances
    
    # Existing initialization steps
    self._train_finetune_init()
    support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)
    self._train_finetune(support_pos_rep)

    # Track existing instances
    exist_id = {}
    
    for snowball_iter in range(snowball_max_iter):
        # Phase 1: Enhanced Expansion with Top-k Approach
        # Collect entity pairs from support set
        entpair_support = {}
        for i in range(len(support_pos['id'])):
            entpair = support_pos['entpair'][i]
            exist_id[support_pos['id'][i]] = 1
            
            # Initialize entity pair data structure
            if entpair not in entpair_support:
                entpair_support[entpair] = {
                    'word': [], 
                    'mask': [], 
                    'id': [],
                    'entpair': [],
                    'pos1': [] if 'pos1' in support_pos else None,
                    'pos2': [] if 'pos2' in support_pos else None
                }
            
            # Add instance to entity pair
            self._add_ins_to_data(entpair_support[entpair], support_pos, i)
        
        # Top-k Expansion Process
        for entpair, support_data in entpair_support.items():
            # Fetch distant instances with same entity pair
            distant_data = distant.get_same_entpair_ins(entpair)
            if distant_data is None:
                continue
            
            # Prepare candidate instances
            candidates = []
            for i in range(distant_data['word'].size(0)):
                if distant_data['id'][i] not in exist_id:
                    candidates.append(i)
            
            # If no candidates, continue to next entity pair
            if not candidates:
                continue
            
            # Prepare candidate dataset
            candidate_dataset = {}
            for key in distant_data:
                if isinstance(distant_data[key], torch.Tensor):
                    candidate_dataset[key] = distant_data[key][candidates]
                else:
                    candidate_dataset[key] = [distant_data[key][j] for j in candidates]
            
            # Stack and move to cuda
            self._dataset_stack_and_cuda(support_data)
            self._dataset_stack_and_cuda(candidate_dataset)
            
            # Get top-k similar instances
            pick_or_not = self.siamese_model.forward_infer_sort(
                support_data, 
                candidate_dataset, 
                batch_size=self.args.infer_batch_size
            )
            
            # Select top-k instances based on similarity
            added_count = 0
            for i in range(min(len(pick_or_not), top_k)):
                if pick_or_not[i][0] > self.args.phase1_siamese_th:
                    iid = pick_or_not[i][1]
                    self._add_ins_to_vdata(support_pos, candidate_dataset, iid, label=1)
                    exist_id[candidate_dataset['id'][iid]] = 1
                    added_count += 1
            
            # Update metrics
            self._phase1_add_num = added_count
            self._phase1_total = len(candidates)
        
        # Re-encode support set after expansion
        support_pos_rep = self.encode(support_pos, batch_size=self.args.infer_batch_size)
        
        # Fine-tune with new support set
        self._train_finetune_init()
        self._train_finetune(support_pos_rep)
        
        # Phase 2: Random Candidate Selection with Top-k
        candidate = distant.get_random_candidate(
            self.pos_class, 
            candidate_num_class, 
            candidate_num_ins_per_class
        )
        
        # Classify candidates
        candidate_prob = self._infer(candidate, batch_size=self.args.infer_batch_size)
        
        # Use Siamese model for similarity
        pick_or_not = self.siamese_model.forward_infer_sort(
            support_pos, 
            candidate, 
            batch_size=self.args.infer_batch_size
        )
        
        # Select top-k instances
        added_count = 0
        for i in range(min(len(pick_or_not), top_k)):
            iid = pick_or_not[i][1]
            if (pick_or_not[i][0] > self.args.phase2_siamese_th and 
                candidate_prob[iid] > self.args.phase2_cl_th and 
                candidate['id'][iid] not in exist_id):
                
                exist_id[candidate['id'][iid]] = 1
                self._add_ins_to_vdata(support_pos, candidate, iid, label=1)
                added_count += 1
        
        # Update phase 2 metrics
        self._phase2_add_num = added_count
        self._phase2_total = candidate['word'].size(0)
        
        # Final evaluation
        support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)
        self._train_finetune_init()
        self._train_finetune(support_pos_rep)
        
        # Evaluate on query set
        if self.args.eval:
            self._forward_eval_binary(query, threshold)
    
    # Final evaluation
    self._forward_eval_binary(query, threshold)
    
    return support_pos_rep
    def _forward_eval_binary(self, query, threshold=0.5):
        '''
        snowball process (eval)
        query: query set (raw data)
        threshold: ins with prob > threshold will be classified as positive
        return (accuracy at threshold, precision at threshold, recall at threshold, f1 at threshold, auc), 
        '''
        query_prob = self._infer(query, batch_size=self.args.infer_batch_size).cpu().detach().numpy()
        label = query['label'].cpu().detach().numpy()
        accuracy = float(np.logical_or(np.logical_and(query_prob > threshold, label == 1), np.logical_and(query_prob < threshold, label == 0)).sum()) / float(query_prob.shape[0])
        if (query_prob > threshold).sum() == 0:
            precision = 0
        else:
            precision = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((query_prob > threshold).sum())
        recall = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((label == 1).sum())
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = float(2.0 * precision * recall) / float(precision + recall)
        auc = sklearn.metrics.roc_auc_score(label, query_prob)
        if self.args.print_debug:
            print('')
            sys.stdout.write('[EVAL] acc: {0:2.2f}%, prec: {1:2.2f}%, rec: {2:2.2f}%, f1: {3:1.3f}, auc: {4:1.3f}'.format(\
                    accuracy * 100, precision * 100, recall * 100, f1, auc) + '\r')
            sys.stdout.flush()
        self._accuracy = accuracy
        self._prec = precision
        self._recall = recall
        self._f1 = f1
        return (accuracy, precision, recall, f1, auc)

    def forward(self, support_pos, query, distant, pos_class, threshold=0.5, threshold_for_snowball=0.5):
        '''
        snowball process (train + eval)
        support_pos: support set (positive, raw data)
        support_neg: support set (negative, raw data)
        query: query set (raw data)
        distant: distant data loader
        pos_class: positive relation (name)
        threshold: ins with prob > threshold will be classified as positive
        threshold_for_snowball: distant ins with prob > th_for_snowball will be added to extended support set
        '''
        self.pos_class = pos_class 

        self._forward_train(support_pos, query, distant, threshold=threshold)

    def init_10shot(self, Ws, bs):
        self.Ws = torch.stack(Ws, 0).transpose(0, 1) # (230, 16)
        self.bs = torch.stack(bs, 0).transpose(0, 1) # (1, 16)

    def eval_10shot(self, query):
        x = self.sentence_encoder(query)
        x = torch.matmul(x, self.Ws) + self.new_bias # (batch_size, 16)
        x = torch.sigmoid(x)
        _, pred = x.max(-1) # (batch_size)
        return self.__accuracy__(pred, query['label'])

