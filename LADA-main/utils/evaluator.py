import numpy as np
from collections import OrderedDict, defaultdict
import torch


class Evaluator:
    """Evaluator for classification."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset([0])

    def reset(self, indices):
        self.indices = indices
        self.indices_tensor = torch.tensor(self.indices)
        
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_conf = []  # Store prediction confidences
        
        self._task_pred = []
        self._task_true = []

    def process(self, mo, gt):
        # mo: model output [batch, num_classes]
        # gt: ground truth [batch]
        pred = mo.max(1)[1]
        conf = torch.softmax(mo, dim=1).max(1)[0]  # Compute prediction confidences
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())
        self._y_conf.extend(conf.data.cpu().numpy().tolist())

        task_truth = (gt.unsqueeze(1).cpu() >= self.indices_tensor).int().sum(dim=1) - 1
        task_pred = (pred.unsqueeze(1).cpu() >= self.indices_tensor).int().sum(dim=1) - 1

        self._task_true.extend(task_truth.data.cpu().numpy().tolist())
        self._task_pred.extend(task_pred.data.cpu().numpy().tolist())

    def evaluate(self):
        indices = self.indices
        results = OrderedDict()

        self._task_selection_recall = defaultdict(list)
        self._task_selection_precision = defaultdict(list)

        for label, pred in zip(self._task_true, self._task_pred):
            matches = int(label == pred)
            self._task_selection_recall[label].append(matches)
            self._task_selection_precision[pred].append(matches)

        task_id = list(self._task_selection_recall.keys())
        task_id.sort()

        task_selection_recall = []
        task_selection_precision = []
        for id in task_id:
            res = self._task_selection_recall[id]
            correct = sum(res)
            total = len(res)
            task_selection_recall.append(correct / total * 100)
            res = self._task_selection_precision[id]
            correct = sum(res)
            total = len(res)
            task_selection_precision.append(correct / total * 100)

        for id in task_id:
            print(f"* Task {id} selection Recall: {task_selection_recall[id]:.1f}% | Precision: {task_selection_precision[id]:.1f}%")
        print(f"* Average Task selection Recall: {np.mean(task_selection_recall):.1f}% | Precision: {np.mean(task_selection_precision):.1f}%\n")

        self._per_class_res = defaultdict(list)

        for label, pred in zip(self._y_true, self._y_pred):
            matches = int(label == pred)
            self._per_class_res[label].append(matches)

        labels = list(self._per_class_res.keys())
        labels.sort()

        cls_correct = []
        cls_total = []
        cls_accs = []
        for label in labels:
            res = self._per_class_res[label]
            correct = sum(res)
            cls_correct.append(correct)
            total = len(res)
            cls_total.append(total)
            acc = 100.0 * correct / total
            cls_accs.append(acc)
        
        cls_correct = np.array(cls_correct)
        cls_total = np.array(cls_total)
        acc_list = []

        for i in range(len(indices)):
            if i != len(indices) - 1:
                acc_list.append(np.sum(cls_correct[indices[i]:indices[i+1]]) / np.sum(cls_total[indices[i]:indices[i+1]]) * 100)
            else:
                acc_list.append(np.sum(cls_correct[indices[i]:]) / np.sum(cls_total[indices[i]:]) * 100)

        for i in range(len(acc_list)):
            print(f"* Task {i} Accuracy: {acc_list[i]:.1f}%")
        print(f"* Average Accuracy: {np.mean(acc_list):.1f}%")
        
        return results

