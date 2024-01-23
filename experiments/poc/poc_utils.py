import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
import evaluate

def compute_metrics(p, tokenizer):
    rouge = evaluate.load('rouge')
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = rouge.compute(predictions=predictions, references=labels)
    results = {k: np.mean(v) for k, v in results.items()}
    return results


def collate_fn(batch, tokenizer, max_length):
    text_inputs = [("revise: summary: " + row['summary'], " document: " + row['document']) for row in batch]
    revised_summaries = [row['revised_summary'] for row in batch]
    inputs = tokenizer.batch_encode_plus(text_inputs, padding=True, truncation='only_second', max_length=max_length,
                                         return_tensors='pt')
    labels = tokenizer(revised_summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}




def load_xsum_ood(only_low_score=False, num_of_examples=1000):
    ood_test_set = load_dataset('xsum', split='test')
    low_score_indices = [3, 10, 13, 14, 15, 17, 18, 20, 23, 25, 28, 30, 32, 33, 38, 39, 40, 42, 44, 47, 48, 49, 53,
                         54,
                         60, 63, 67, 72, 76, 80, 81, 87, 89, 90, 92, 97, 98, 99, 101, 102, 105, 107, 114, 115, 116,
                         119,
                         120, 121, 124, 127, 128, 130, 131, 135, 138, 139, 141, 142, 144, 146, 148, 149, 151, 152,
                         153,
                         154, 157, 158, 160, 161, 166, 167, 172, 174, 175, 180, 183, 185, 192, 195, 197, 198, 202,
                         204,
                         205, 209, 220, 223, 225, 229, 236, 237, 238, 243, 244, 253, 255, 262, 265, 267, 269, 272,
                         277,
                         279, 280, 281, 284, 289, 291, 295, 297, 299, 304, 306, 307, 309, 310, 316, 322, 323, 325,
                         329,
                         332, 333, 336, 337, 344, 346, 347, 348, 351, 353, 354, 356, 357, 358, 362, 365, 366, 370,
                         371,
                         373, 374, 375, 377, 379, 382, 385, 386, 390, 391, 392, 393, 396, 402, 405, 406, 407, 413,
                         414,
                         415, 420, 421, 423, 427, 431, 433, 440, 441, 442, 447, 449, 450, 451, 459, 460, 462, 468,
                         469,
                         470, 471, 473, 474, 479, 481, 482, 484, 485, 486, 487, 492, 493, 494, 495, 496, 497, 502,
                         505,
                         507, 512, 516, 520, 527, 530, 532, 534, 541, 544, 547, 551, 553, 554, 560, 562, 563, 564,
                         570,
                         571, 577, 578, 579, 582, 585, 587, 589, 590, 593, 594, 602, 603, 609, 611, 617, 620, 622,
                         624,
                         626, 629, 632, 641, 642, 643, 646, 649, 650, 653, 655, 657, 658, 671, 672, 676, 677, 678,
                         679,
                         689, 691, 695, 696, 698, 706, 709, 713, 721, 723, 724, 725, 726, 727, 729, 730, 732, 734,
                         736,
                         740, 745, 748, 751, 757, 759, 766, 768, 770, 771, 777, 780, 782, 783, 784, 785, 788, 791,
                         792,
                         793, 796, 799, 801, 803, 804, 808, 809, 810, 811, 813, 814, 817, 818, 822, 823, 825, 828,
                         830,
                         831, 833, 834, 835, 839, 841, 846, 847, 853, 856, 857, 860, 861, 862, 863, 864, 865, 870,
                         876,
                         878, 879, 880, 881, 882, 887, 890, 892, 893, 898, 899, 901, 902, 903, 904, 905, 912, 913,
                         914,
                         915, 920, 921, 922, 925, 926, 927, 930, 931, 935, 938, 941, 943, 950, 951, 954, 955, 957,
                         958,
                         960, 963, 965, 966, 967, 968, 971, 972, 975, 980, 987, 990, 994, 995, 996, 997, 1000, 1002,
                         1005, 1006, 1008, 1009, 1012, 1016, 1018, 1024, 1029, 1038, 1041, 1042, 1044, 1046, 1047,
                         1049,
                         1050, 1051, 1053, 1054, 1055, 1056, 1057, 1060, 1061, 1062, 1065, 1066, 1075, 1077, 1081,
                         1082,
                         1089, 1092, 1094, 1095, 1100, 1103, 1106, 1109, 1113, 1116, 1119, 1126, 1129, 1135, 1138,
                         1140,
                         1146, 1149, 1150, 1151, 1152, 1153, 1159, 1162, 1166, 1170, 1173, 1176, 1177, 1179, 1181,
                         1183,
                         1185, 1186, 1191, 1193, 1194, 1199, 1200, 1206, 1207, 1209, 1212, 1215, 1220, 1225, 1227,
                         1232,
                         1233, 1234, 1237, 1238, 1240, 1241, 1243, 1247, 1248, 1249, 1251, 1252, 1253, 1255, 1258,
                         1260,
                         1263, 1269, 1270, 1271, 1272, 1274, 1283, 1284, 1288, 1289, 1290, 1292, 1293, 1300, 1301,
                         1304,
                         1305, 1318, 1324, 1326, 1330, 1334, 1336, 1342, 1344, 1353, 1355, 1360, 1362, 1363, 1365,
                         1366,
                         1367, 1372, 1376, 1380, 1381, 1385, 1390, 1393, 1398, 1399, 1405, 1408, 1410, 1412, 1413,
                         1415,
                         1416, 1419, 1421, 1429, 1431, 1432, 1434, 1437, 1441, 1446, 1452, 1453, 1455, 1458, 1459,
                         1460,
                         1463, 1464, 1467, 1468, 1469, 1470, 1471, 1472, 1474, 1475, 1479, 1480, 1481, 1482, 1483,
                         1484,
                         1488, 1493, 1494, 1496]
    if only_low_score:
        ood_test_texts = [ood_test_set[i]['document'] for i in low_score_indices]
        ood_test_summaries = [ood_test_set[i]['summary'] for i in low_score_indices]
    else:
        ood_test_texts = [ood_test_set[i]['document'] for i in range(num_of_examples)]
        ood_test_summaries = [ood_test_set[i]['summary'] for i in range(num_of_examples)]
    return ood_test_texts, ood_test_summaries
