import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import data_process
import numpy as np
import plot_scorefunc
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
class GMZ(torch.nn.Module):
    def __init__(self, input1,input2, **kwargs):  # input1: gamma * num of attributes, input2 : num of criteria
        super(GMZ, self).__init__()
        self.layer1 = torch.nn.Conv1d(input1,input1,1,1,0,bias=0,groups=input1) # group = number of criteria
        self.layer2 = torch.nn.Conv1d(input2,input2,1,1,0,bias=0,groups=input2)
        self.layer3_2 = torch.nn.Sequential(
            torch.nn.Linear(input2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,1)
        )
        self.layer_weight = torch.nn.Sequential(
            torch.nn.Linear(input1, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,1),
            torch.nn.Sigmoid()
        )
        self.layer_sig = torch.nn.Sigmoid() # For binary classification

    def forward(self, input_linear, input_nonlinear, mean, K, train_size, num_criteria, test_weight=None):
        y1 = self.layer1(input_linear) # First layer in linear part: correspond to each marginal score
        # print(y1.shape)
        index = np.linspace(0, K * num_criteria, K * num_criteria, endpoint=False, dtype=int)
        temp_y1 = torch.zeros(size=(train_size, num_criteria, 1))
        for i in range(0, K):
            temp_index = index[i::K]
            temp_y1 += y1[:, temp_index, :]
        y1 = temp_y1 # Sum all pieces for each attribute
        #############
        #############
        # For each attribute, we give an individual weights
        y2 = self.layer2(y1)
        # Parameters in layer2 are weights for each attribute?
        linear_out = torch.sum(y2, dim=1)
        # Sum all marginal scores, the output of linear part
        #############
        # For nonlinear part, the input is the original attribute values
        #############
        nonlinear_out = self.layer3_2(input_nonlinear)
        #############
        # The tradeoff coefficient, range 0-1
        if test_weight is None:
            # weight = self.layer_weight(mean)
            weight = torch.FloatTensor(np.array(0.5))
        else:
            weight = test_weight
        # weight = torch.FloatTensor(np.array(1.0))

        link_out = linear_out * weight + nonlinear_out * (1 - weight)
        # link_out = linear_out  + nonlinear_out
        # By Sigmoid, link_out is bineary classification
        # By sofmax, link_out is multiple classification
        # By identity, link_out is regression
        # link_out = self.layer_sig(link_out)
        return link_out, weight

def LS(Phi,samp_y): #calculate LeastSquare Regression
    Y = samp_y.reshape(-1, 1)  # reshape Y to 50*1
    # based on the equation
    Phi = Phi.T
    theta_LS = np.dot(np.dot(np.linalg.inv(np.dot(Phi,Phi.T)),Phi),Y)
    return theta_LS

def RLS(Phi,samp_y,lamd): #calculate Regularized LS Regression
    Y = samp_y.reshape(-1, 1)
    Phi = Phi.T
    # theta_RLS = np.dot(np.dot( np.linalg.inv(np.dot(Phi, Phi.T) + lamd*np.ones((Phi.shape[0],1)) ), Phi) , Y)
    theta_RLS = np.dot(np.dot( np.linalg.inv(np.dot(Phi, Phi.T) + lamd*np.eye((Phi.shape[0])) ), Phi) , Y)
    return theta_RLS


if __name__ == '__main__':
    np.random.seed(70) # 69 70 1000
    BATCH_SIZE = 2000
    number_of_attr = 50
    number_of_object = 10000
    gamma = 3 # predefined number of pieces
    degree = 10

    Ori_Data = data_process.generate_attribute_value(number_of_object, number_of_attr)
    # For regression
    New_Data, global_value, coeff = data_process.marginalvalue_with_interaction(Ori_Data, degree=degree, pair_num=900)
    # New_Data, global_value, coeff = data_process.marginalvalue(Ori_Data, degree=degree)
    # For classification, class_num=2 for binary
    # New_Data, global_value, coeff = data_process.marginalvalue_with_interaction_classifcation(Ori_Data, degree=degree,class_num=2, pair_num=None)

    interval_vector, X_train, X_test, Y_train, Y_test = data_process.piecewise_linear('./simulation_data.csv', './global_value.csv',
                                                                         gamma, 0.2) # New_Data, X_train, Y_train are Vectorized
    # interval_vector, X_train, X_test, Y_train, Y_test = data_process.piecewise_linear('./bank_additional_full_preprocess.csv', './OnlineNewsLabel.csv', gamma, 0.2)
    # interval_vector, X_train, X_test, Y_train, Y_test = data_process.piecewise_linear('./bank-additional-full-preprocess-centralized-1-numerical.csv', './labels.csv', gamma, 0.2)

    # Intialization for regression
    # theta = RLS(X_train[:,0:-number_of_attr], Y_train, 0.01)
    theta = LS(X_train[:,0:-number_of_attr], Y_train)
    print(theta)
    print(len(theta))
    pred_y_train = np.dot(X_train[:,0:-number_of_attr], theta)
    # print(pred_y_train)
    # print(Y_train)
    print(np.square(np.ravel(pred_y_train) - Y_train).mean())
    pred_y = np.dot(X_test[:,0:-number_of_attr], theta)
    # print(pred_y)
    # print(Y_test - np.ravel(pred_y))
    # print(np.square(Y_test-np.ravel(pred_y)))
    # print(np.mean(np.square(Y_test-np.ravel(pred_y))))
    print(np.square(Y_test - np.ravel(pred_y)).mean())

    # Initialization for classfication
    # clf = LogisticRegression(C=0.01, fit_intercept=True,max_iter=100).fit(X_train[:,0:-number_of_attr], Y_train)
    # pred_y = clf.predict_proba(X_test[:,0:-number_of_attr])[:,1]
    # theta = clf.coef_
    # print((clf.coef_).shape)
    # print(roc_auc_score(Y_test, pred_y))
    # plot_scorefunc.plot_func(num_of_attr=number_of_attr, interval_vector=interval_vector,w=np.ravel(theta))
    plot_scorefunc.plot_ploy_func(number_of_attr, interval_vector, np.ravel(theta), coeff, degree)
    # For binary classification
    # print('This is AUC scores for pretraining: ***************************************')
    # print(roc_auc_score(Y_test, pred_y))
    torch_dataset = Data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # random shuffle for training
        num_workers=0,  # subprocesses for loading data
    )
    # Initialization for tradeoff coefficients
    mean = np.mean(X_train[:, 0:-number_of_attr], axis=0)
    mean = torch.from_numpy(mean)
    mean = mean.type(torch.FloatTensor)

    # Initialization for Network

    # theta = torch.from_numpy(np.ravel(theta)[:gamma*number_of_attr])
    # theta = theta.type(torch.FloatTensor)
    theta = np.squeeze(theta[:gamma*number_of_attr])
    # print(theta.shape)
    theta = torch.from_numpy(theta)
    theta = theta.type(torch.FloatTensor)
    theta = theta.view(-1, 1 ,1)
    # theta = torch.squeeze(theta,0)
    # print(theta.size)
    # theta = theta.type(torch.FloatTensor)
    # theta.view(number_of_attr * gamma,-1, 1)
    net = GMZ(number_of_attr * gamma, number_of_attr)

    for name, params in net.named_parameters():
        # print(name)
        # print(params)
        if name == 'layer1.weight':
            # print('OOOOOO')
            # params.requires_grad = False
            params.data = theta
            # print(params)
            # name.weight
    print(net)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, betas=(0.9, 0.999)) # NOT train linear part
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.008)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    # Regression Loss
    loss_func = torch.nn.MSELoss()
    # Check params
    params = []
    for param in net.parameters():
        params.append(param.detach().numpy())
    # print(np.array(params))
    parameters = []
    for item in np.array(params):
        item = np.ravel(item)
        parameters.append(item)
    # print(parameters[0])
    # print(theta)

    for epoch in range(150):  # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data...
            # Input of Linear part should be Vectorized data: the COL from the first to LAST number of attributes
            # Input of Nonlinear DL part should be ORIGINAL data: the COL from the LAST to LAST number of attributes
            # Instance: number of attribute: 20, Gamma: 5 --> Vectorized data: batch_x[:, 0:-20]; Original: batch_x[:,-20:]

            input_linear = batch_x[:, 0:-number_of_attr]
            input_linear = input_linear.type(torch.FloatTensor)
            input_nonlinear = batch_x[:, -number_of_attr:]
            input_nonlinear = input_nonlinear.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_y = batch_y.view(-1, 1)
            train_size = batch_x.shape[0]
            # transform input data
            input_linear = input_linear.view(train_size, -1, 1)
            # input_nonlinear = input_nonlinear.view(train_size, -1, 1)
            # print(input_linear.shape, input_nonlinear.shape)
            # Train Net
            prediction, weight = net(input_linear, input_nonlinear, mean, gamma, train_size, number_of_attr,
                                      torch.FloatTensor(np.array(0.50)))
            params = []
            for param in net.parameters():
                params.append(param)
            # reg_loss = torch.sum(torch.mul(params[0], params[0]))  # L2 Regularization
            reg_loss = torch.sum(torch.abs(params[0]))
            # loss = loss_func(prediction, batch_y)+reg_loss
            loss = loss_func(prediction, batch_y)
            real_loss = loss_func(prediction, batch_y)
            # print(roc_auc_score(batch_y,prediction.detach().numpy()))
            # loss = loss + (1 - weight)**2 # Regularization
            optimizer.zero_grad()  # 参数梯度降为0
            loss.backward()  # 反向传递
            optimizer.step()  # 优化梯度
        print('Epoch: ', epoch, )
        print('********************')
        print('| Step: ', step, )
        print(loss)
        print(real_loss)
        print(weight)

    torch.save(net, 'net_params_aft_pretrain.pkl')
    params = []
    for param in net.parameters():
        params.append(param.detach().numpy())
    # print(np.array(params))
    parameters = []
    for item in np.array(params):
        item = np.ravel(item)
        parameters.append(item)
    print(parameters[0])
    plot_scorefunc.plot_ploy_func(number_of_attr, interval_vector, parameters[0], coeff, degree)
    # plot_scorefunc.plot_func(number_of_attr, interval_vector, parameters[0])
    net3 = torch.load('net_params_aft_pretrain.pkl')
    test_size = X_test.shape[0]
    # linear and nonlinear input parts
    input_linear = X_test[:, 0:-number_of_attr]
    input_linear = torch.from_numpy(input_linear).reshape((test_size, -1, 1))
    input_linear = input_linear.type(torch.FloatTensor)
    input_nonlinear = X_test[:, -number_of_attr:]
    input_nonlinear = torch.from_numpy(input_nonlinear)
    input_nonlinear = input_nonlinear.type(torch.FloatTensor)
    Y_test = torch.from_numpy(Y_test).reshape((-1, 1))
    Y_test = Y_test.type(torch.FloatTensor)
    # Get test
    prediction, weight = net3(input_linear, input_nonlinear, mean, gamma, test_size, number_of_attr, weight)
    loss_func = torch.nn.MSELoss()
    loss_test = loss_func(prediction, Y_test)
    print(loss_test, weight)
    # pred_y = prediction.detach().numpy()
    # Y_test = Y_test.detach().numpy()
    # print(np.square(np.ravel(Y_test) - np.ravel(pred_y)).mean())


    # print('This is AUC scores for final model: ***************************************')
    # print(roc_auc_score(Y_test, prediction.detach().numpy()))
    # plot_scorefunc.plot_func(number_of_attr, interval_vector, parameters[0])
    # plot_scorefunc.plot_poly(coeff, 3, 20)