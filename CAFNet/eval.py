import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

path = './tmp1/checkpoint_epoch_74.pt'   # the path of the model
model = torch.load(path)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()

def eval_qwk_lgb_regr(y_true, y_pred):
  # Fast cappa eval function for lgb.
    dist = Counter(reduce_train['accuracy_group'])
    for k in dist:
        dist[k] /= len(reduce_train)
    reduce_train['accuracy_group'].hist()
    # reduce_train['accuracy_group']将会分成四组
    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_pred, acum * 100)

    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)
    return 'cappa', cohen_kappa_score(y_true, y_pred, weights='quadratic'), True

with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)
        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        # tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
        #                 cd_preds.data.cpu().numpy().flatten()).ravel()
        try:
            tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),cd_preds.data.cpu().numpy().flatten()).ravel()
        except:
            # tn = confusion_matrix(labels.data.cpu().numpy().flatten(),cd_preds.data.cpu().numpy().flatten()).ravel()
            # fp, fn, tp = 0, 0, 0
            continue

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
# PCC = (fp + fn)/(fp + fn + tp + tn)
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)
PCC = (tp+tn)/(tp+tn+fp+fn)
Pe = ((tp+fn)*(tp+fp)+(fp+tn)*(fn+tn))/((tp+tn+fp+fn)*(tp+tn+fp+fn))
KC = (PCC-Pe)/(1-Pe)
IOU = (R * R)/(P+R-(P*R))
print('Precision: {}\nRecall: {}\nF1-Score: {}\nPCC: {}\nKC: {}\nIOU: {}'.format(P, R, F1,PCC,KC,IOU))
