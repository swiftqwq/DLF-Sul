import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn import preprocessing
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
import math
import pandas as pd


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode(p_string, n_string):
    n = len(p_string)
    m = len(n_string)
    m_l = len(p_string[0])
    x_BE_p = np.zeros((n, m_l, 20))
    x_62_p = np.zeros((n, m_l, 20))
    x_AAI_p = np.zeros((n, m_l, 14))
    x_BE_n = np.zeros((m, m_l, 20))
    x_62_n = np.zeros((m, m_l, 20))
    x_AAI_n = np.zeros((m, m_l, 14))
    maxabs = preprocessing.MinMaxScaler()
    for i in range(n):
        x_BE_p[i], x_62_p[i], x_AAI_p[i] = forward(p_string[i])
    for i in range(m):
        x_BE_n[i], x_62_n[i], x_AAI_n[i] = forward(n_string[i])
    # print(x_AAI[0])
    x_BE_p = x_BE_p.reshape(-1, 1)
    x_62_p = x_62_p.reshape(-1, 1)
    x_AAI_p = x_AAI_p.reshape(-1, 1)
    x_BE_n = x_BE_n.reshape(-1, 1)
    x_62_n = x_62_n.reshape(-1, 1)
    x_AAI_n = x_AAI_n.reshape(-1, 1)
    x_BE_p = maxabs.fit_transform(x_BE_p)
    x_62_p = maxabs.fit_transform(x_62_p)
    x_AAI_p = maxabs.fit_transform(x_AAI_p)
    x_BE_n = maxabs.fit_transform(x_BE_n)
    x_62_n = maxabs.fit_transform(x_62_n)
    x_AAI_n = maxabs.fit_transform(x_AAI_n)
    x_BE_p = x_BE_p.reshape(n, m_l, -1)
    x_62_p = x_62_p.reshape(n, m_l, -1)
    x_AAI_p = x_AAI_p.reshape(n, m_l, -1)
    x_BE_n = x_BE_n.reshape(m, m_l, -1)
    x_62_n = x_62_n.reshape(m, m_l, -1)
    x_AAI_n = x_AAI_n.reshape(m, m_l, -1)
    x_BE_p = torch.tensor(x_BE_p, dtype=torch.float32, requires_grad=True)
    x_62_p = torch.tensor(x_62_p, dtype=torch.float32, requires_grad=True)
    x_AAI_p = torch.tensor(x_AAI_p, dtype=torch.float32, requires_grad=True)
    x_BE_n = torch.tensor(x_BE_n, dtype=torch.float32, requires_grad=True)
    x_62_n = torch.tensor(x_62_n, dtype=torch.float32, requires_grad=True)
    x_AAI_n = torch.tensor(x_AAI_n, dtype=torch.float32, requires_grad=True)
    input_batch_p = torch.cat((x_BE_p, x_62_p, x_AAI_p), 2)
    input_batch_n = torch.cat((x_BE_n, x_62_n, x_AAI_n), 2)
    # print(x_BE_p.shape, x_62_p.shape, x_AAI_p.shape)
    # print(input_batch_p.shape)
    return input_batch_p, input_batch_n


def forward(x):
    x_BE = BE(x)  # (len,20)
    x_62 = BLOSUM62(x)  # (len,20)
    x_AAI = AAI(x)  # (len,14)
    return x_BE, x_62, x_AAI


def BE(gene):
    with open("BE.txt") as f:
        records = f.readlines()[1:]
    BE = []
    for i in records:
        array = i.rstrip().split() if i.rstrip() != "" else None
        BE.append(array)
    BE = np.array(
        [float(BE[i][j]) for i in range(len(BE)) for j in range(len(BE[i]))]
    ).reshape((20, 21))
    BE = BE.transpose()
    AA = "ACDEFGHIKLMNPQRSTWYV*"
    GENE_BE = {}
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)
    gene_array = np.zeros((n, 20))
    for i in range(n):
        gene_array[i] = BE[(GENE_BE[gene[i]])]
    return gene_array


def BLOSUM62(gene):
    with open("blosum62.txt") as f:
        records = f.readlines()[1:]
    blosum62 = []
    for i in records:
        array = i.rstrip().split() if i.rstrip() != "" else None
        blosum62.append(array)
    blosum62 = np.array(
        [
            float(blosum62[i][j])
            for i in range(len(blosum62))
            for j in range(len(blosum62[i]))
        ]
    ).reshape((20, 21))
    blosum62 = blosum62.transpose()
    GENE_BE = {}
    AA = "ARNDCQEGHILKMFPSTWYV*"
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)
    gene_array = np.zeros((n, 20))
    for i in range(n):
        gene_array[i] = blosum62[(GENE_BE[gene[i]])]
    return gene_array


def AAI(gene):
    with open("AAI.txt") as f:
        records = f.readlines()[1:]
    AAI = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != "" else None
        AAI.append(array)
    AAI = np.array(
        [float(AAI[i][j]) for i in range(len(AAI)) for j in range(len(AAI[i]))]
    ).reshape((14, 21))
    AAI = AAI.transpose()
    GENE_BE = {}
    AA = "ACDEFGHIKLMNPQRSTWYV*"
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)
    gene_array = np.zeros((n, 14))
    for i in range(n):
        gene_array[i] = AAI[(GENE_BE[gene[i]])]
    return gene_array


def train_data():
    target_batch_p = []
    target_batch_n = []
    with open("train_p.txt", "r") as f:
        train_p_string = f.read().split("\n")
    with open("train_n2.txt", "r") as f:
        train_n_string = f.read().split("\n")
    input_batch_p, input_batch_n = encode(train_p_string, train_n_string)
    for i in range(len(train_p_string)):
        target_batch_p.append([1, 0])
    for i in range(len(train_n_string)):
        target_batch_n.append([0, 1])
    # print(input_batch)
    # print(target_batch)
    return (
        torch.FloatTensor(input_batch_p),
        torch.FloatTensor(input_batch_n),
        torch.FloatTensor(target_batch_p),
        torch.FloatTensor(target_batch_n),
    )


def validate_data():
    target_batch_p = []
    target_batch_n = []
    with open("validation_p.txt", "r") as f:
        validate_p_string = f.read().split("\n")
    with open("validation_n.txt", "r") as f:
        validate_n_string = f.read().split("\n")
    input_batch_p, input_batch_n = encode(validate_p_string, validate_n_string)
    for i in range(len(validate_p_string)):
        target_batch_p.append([1, 0])
    for i in range(len(validate_n_string)):
        target_batch_n.append([0, 1])
    return (
        torch.FloatTensor(input_batch_p),
        torch.FloatTensor(input_batch_n),
        torch.FloatTensor(target_batch_p),
        torch.FloatTensor(target_batch_n),
    )


def test_data():
    target_batch_p = []
    target_batch_n = []
    with open("test_p.txt", "r") as f:
        test_p_string = f.read().split("\n")
    with open("test_n4.txt", "r") as f:
        test_n_string = f.read().split("\n")
    input_batch_p, input_batch_n = encode(test_p_string, test_n_string)
    for i in range(len(test_p_string)):
        target_batch_p.append([1, 0])
    for i in range(len(test_n_string)):
        target_batch_n.append([0, 1])
    return (
        torch.FloatTensor(input_batch_p),
        torch.FloatTensor(input_batch_n),
        torch.FloatTensor(target_batch_p),
        torch.FloatTensor(target_batch_n),
    )


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.blstm1 = nn.LSTM(
            input_size=20,
            hidden_size=n_hidden,
            bidirectional=True,
            batch_first=True,
            device=device,
        )
        self.blstm2 = nn.LSTM(
            input_size=20,
            hidden_size=n_hidden,
            bidirectional=True,
            batch_first=True,
            device=device,
        )
        self.blstm3 = nn.LSTM(
            input_size=14,
            hidden_size=n_hidden,
            bidirectional=True,
            batch_first=True,
            device=device,
        )
        self.W_Q = nn.Linear(195, d_k * n_heads, bias=False, device=device)
        self.W_K = nn.Linear(195, d_k * n_heads, bias=False, device=device)
        self.W_V = nn.Linear(195, d_v * n_heads, bias=False, device=device)
        self.fc = nn.Sequential(
            nn.Linear(n_heads * d_v, 195, bias=False, device=device)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=12, kernel_size=6, stride=2, device=device
            ),
            nn.BatchNorm2d(12, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=12, kernel_size=6, stride=2, device=device
            ),
            nn.BatchNorm2d(12, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=12, kernel_size=6, stride=2, device=device
            ),
            nn.BatchNorm2d(12, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.FC = nn.Sequential(
            nn.Linear(7020, 64, device=device),
            nn.Dropout(0.5),
            nn.Linear(64, 2, device=device),
            torch.nn.Sigmoid(),
        )

    def attention(self, input_Q, input_K, input_V, d_model):
        residual, batch_size1 = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size1, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size1, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size1, -1, n_heads, d_v).transpose(1, 2)
        # context: [batch_size, n_heads, len_q, d_v]
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        context = context.transpose(1, 2).reshape(batch_size1, -1, n_heads * d_v)
        # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        # print(output.shape)
        # print(residual.shape)
        return nn.LayerNorm(d_model, device=device)(output + residual)

    def forward(self, X):
        # print(X.shape)
        # exit()
        # X: [batch_size, n_step, n_class]
        batch_size = X.shape[0]
        # print(batch_size)
        hidden_state1 = torch.zeros(
            1 * 2, batch_size, n_hidden, device=device
        )  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state1 = torch.zeros(
            1 * 2, batch_size, n_hidden, device=device
        )  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs1, (_, _) = self.blstm1(X[:, :, 0:20], (hidden_state1, cell_state1))

        hidden_state2 = torch.zeros(
            1 * 2, batch_size, n_hidden, device=device
        )  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state2 = torch.zeros(
            1 * 2, batch_size, n_hidden, device=device
        )  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs2, (_, _) = self.blstm2(X[:, :, 20:40], (hidden_state2, cell_state2))

        hidden_state3 = torch.zeros(
            1 * 2, batch_size, n_hidden, device=device
        )  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state3 = torch.zeros(
            1 * 2, batch_size, n_hidden, device=device
        )  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs3, (_, _) = self.blstm3(X[:, :, 40:54], (hidden_state3, cell_state3))

        # print(outputs3.shape)
        # exit()

        x_CNN1_in = outputs1.unsqueeze(1)
        outputs1 = self.conv1(x_CNN1_in)

        x_CNN2_in = outputs2.unsqueeze(1)
        outputs2 = self.conv2(x_CNN2_in)

        x_CNN3_in = outputs3.unsqueeze(1)
        outputs3 = self.conv3(x_CNN3_in)

        outputs = torch.cat((outputs1, outputs2, outputs3), dim=1)
        # print(outputs.shape)
        outputs = outputs.contiguous().view(batch_size, 36, 15 * 13)

        # exit()

        d_model = outputs.size()[-1]
        enc_inputs = outputs.to(device)
        input_Q = enc_inputs
        input_K = enc_inputs
        input_V = enc_inputs
        x_attention_outputs = self.attention(
            enc_inputs, enc_inputs, enc_inputs, d_model
        )
        # print(x_attention_outputs.shape)
        model = self.FC(x_attention_outputs.unsqueeze(1).view(batch_size, -1))
        # print(model.size())
        return model


# Training
def evalute(model, validater):
    model.eval()
    result = []
    label = []
    criterion = nn.BCELoss()
    for x_p, x_n, y_p, y_n in validater:
        # x_p = x_p.cuda()
        # x_n = x_n.cuda()
        x = torch.cat((x_p, x_n), 0).to(device)
        outputs = model(x).cpu()

        target = torch.cat((y_p, y_n), 0)
        loss = criterion(outputs, target)
        # print('loss =', '{:.6f}'.format(loss))
        outputs = outputs[:, 0]
        target = target[:, 0]
        for i in range(len(outputs)):
            if outputs[i] > 0.45:
                outputs[i] = 1.0
            else:
                outputs[i] = 0.0

        for it in outputs:
            result.append(it.item())
        for it in target:
            label.append(it.item())
        val_acc = metrics.accuracy_score(result, label)
    return val_acc


def train():
    best_acc = 0
    besttrain_acc = 0
    result = []
    label = []
    for epoch in tqdm(range(epochs)):
        for x_p, x_n, y_p, y_n in trainloader:
            model.train()
            # print(x_p.shape)
            # print(x)
            x_p = x_p.to(device)
            x_n = x_n.to(device)
            x = torch.cat((x_p, x_n), 0)
            outputs = model(x).cpu()
            # print(pred)
            y = torch.cat((y_p, y_n), 0)
            # print(y)
            # print(outputs.shape, y.shape)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputs = outputs[:, 0]
            target = y[:, 0]
            for i in range(len(outputs)):
                if outputs[i] > 0.45:
                    outputs[i] = 1.0
                else:
                    outputs[i] = 0.0

            for it in outputs:
                result.append(it.item())
            for it in target:
                label.append(it.item())
            if (epoch + 1) % 1 == 0:
                print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.6f}".format(loss))
        train_acc = metrics.accuracy_score(result, label)
        if train_acc > besttrain_acc:
            besttrain_epoch = epoch
            besttrain_acc = train_acc
        val_acc = evalute(model, valloader)
        print(val_acc)
        scheduler.step(val_acc)
        epochs_z.extend([epoch])
        trainacc.extend([train_acc])
        valacc.extend([val_acc])
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(), "models/BLSTM+2头注意力机制+残差+CNN.mdl")
    print("best 训练集acc：", besttrain_acc, "best epoch：", besttrain_epoch)
    print("best 验证集acc：", best_acc, "best epoch：", best_epoch)


def test():
    label_p = []
    result_p = []
    label_n = []
    result_n = []
    Tp = 0
    Tn = 0
    Fp = 0
    Fn = 0
    output_all = []
    target_all = []
    model.eval()
    for x_p, x_n, y_p, y_n in tqdm(testloader):
        # x_p = x_p.cuda()
        # x_n = x_n.cuda()
        x = torch.cat((x_p, x_n), 0).to(device)
        outputs = model(x).cpu()
        target = torch.cat((y_p, y_n), 0)
        outputs_z = torch.split(outputs, int(len(outputs) / 2), dim=0)
        outputs = outputs[:, 0]
        target = target[:, 0]
        outputs_p = outputs_z[0]
        outputs_n = outputs_z[1]
        outputs_p = outputs_p.detach().numpy()
        outputs_n = outputs_n.detach().numpy()
        outputs_p = outputs_p[:, 0]
        outputs_n = outputs_n[:, 0]
        target_p = y_p[:, 0]
        target_n = y_n[:, 0]
        target = target.detach().numpy()
        target_all.extend(target)
        outputs = outputs.detach().numpy()
        output_all.extend(outputs)

        for i in range(len(outputs_p)):
            if outputs_p[i] > 0.45:
                outputs_p[i] = 1.0
            else:
                outputs_p[i] = 0.0
        for i in range(len(outputs_n)):
            if outputs_n[i] > 0.45:
                outputs_n[i] = 1.0
            else:
                outputs_n[i] = 0.0
        for it in outputs_p:
            result_p.append(it.item())
        for it in target_p:
            label_p.append(it.item())
        for it in outputs_n:
            result_n.append(it.item())
        for it in target_n:
            label_n.append(it.item())
        for i in range(len(result_p)):
            if result_p[i] > 0.45:
                Tp = Tp + 1
            else:
                Fp = Fp + 1
        for i in range(len(result_n)):
            if result_n[i] < 0.45:
                Tn = Tn + 1
            else:
                Fn = Fn + 1
    output_all = np.array(output_all)
    target_all = np.array(target_all)
    data1 = pd.DataFrame(output_all)
    data2 = pd.DataFrame(target_all)
    writer1 = pd.ExcelWriter("BLSTM+2头注意力机制+残差+CNN.xlsx")  # 写入Excel文件
    writer2 = pd.ExcelWriter("target.xlsx")  # 写入Excel文件
    data1.to_excel(
        writer1, sheet_name="BLSTM+2头注意力机制+残差+CNN", float_format="%.5f"
    )  # ‘page_1’是写入excel的sheet名
    data2.to_excel(
        writer2, sheet_name="target", float_format="%.1f"
    )  # ‘page_1’是写入excel的sheet名
    # writer1.save()
    # writer2.save()
    writer1.close()
    writer2.close()
    fpr, tpr, thresholds = metrics.roc_curve(target_all, output_all, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    test_acc = (Tp + Tn) / (Tp + Fn + Tn + Fp)
    # test_acc = metrics.accuracy_score(result, label)
    test_Sn = Tp / (Tp + Fn)
    test_Sp = Tn / (Tn + Fp)
    test_Mcc = (Tp * Tn - Fp * Fn) / math.sqrt(
        (Tp + Fn) * (Tp + Fp) * (Tn + Fp) * (Tn + Fn)
    )
    # test_Sn = metrics.accuracy_score(result_p, label_p)
    # test_Sp = metrics.accuracy_score(result_n, label_n)
    print("测试集acc：", test_acc)
    # write test_acc to file test_acc.txt
    with open("test_acc.txt", "w") as f:
        f.write(str(test_acc))
        f.close()
    
    print("Sn：", test_Sn)
    print("Sp：", test_Sp)
    print("Mcc：", test_Mcc)
    print("Auc：", auc)
    plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()


if __name__ == "__main__":
    batch_size = 32
    n_hidden = 32
    d_k = d_v = 195  # dimension of K(=Q), V
    n_heads = 8  # number of heads in Multi-Head Attention
    initial_lr = 0.001
    epochs = 40
    epochs_z = []
    valacc = []
    trainacc = []
    train_batch_p, train_batch_n, train_target_p, train_target_n = train_data()
    val_batch_p, val_batch_n, val_target_p, val_target_n = validate_data()
    test_batch_p, test_batch_n, test_target_p, test_target_n = test_data()
    traindataset = Data.TensorDataset(
        train_batch_p, train_batch_n, train_target_p, train_target_n
    )
    valdataset = Data.TensorDataset(
        val_batch_p, val_batch_n, val_target_p, val_target_n
    )
    testdataset = Data.TensorDataset(
        test_batch_p, test_batch_n, test_target_p, test_target_n
    )

    trainloader = Data.DataLoader(traindataset, batch_size, True)
    valloader = Data.DataLoader(valdataset, batch_size, True)
    testloader = Data.DataLoader(testdataset, batch_size, False)
    model = Network().to(device)
    # model = Network()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        verbose=False,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-08,
    )
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
    train()
    model.load_state_dict(torch.load("models/BLSTM+2头注意力机制+残差+CNN.mdl"))
    #model.load_state_dict(torch.load("models/best2.mdl"))
    test()
    print(trainacc)
    print(valacc)
    plt.axis([0, epochs, 0, 1])
    plt.plot(epochs_z, trainacc, color="b", label="train_acc")
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.title("acc chart")
    plt.legend()
    # plt.show()
    plt.plot(epochs_z, valacc, color="r", label="val_acc")
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.title("acc chart")
    plt.legend()
    plt.savefig("result/BLSTMCNNMutiAttentionCacc.jpg")
    # plt.show()
