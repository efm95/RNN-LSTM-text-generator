from pickletools import optimize
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

class trainer:
    def __init__(self,
                 model,
                 dictionary,
                 lr = 0.001,
                 clip = 1.0):
        
        self.dictionary = dictionary
        self.lr = lr
        self.clip = clip
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr)
        
    def fit(self,
            train_set,
            epochs = 100,
            model_name='model.pt'):
        
        train_x = []
        train_y = []
        
        loss_list = []
        perp_list = []
        count_perp = 0

        for b in range(len(train_set)):
            train_x.append(train_set[b][:-1].transpose(0,1))
            train_y.append(train_set[b][1:].transpose(0,1))
        
        batch_size = train_x[0].shape[0]
        
        self.model.train()
        for epoch in range(epochs):
            
            h = self.model.init_state()
            
            running_loss = 0
            
            for i in range(len(train_x)):
                
                inputs = train_x[i].to(self.device)
                targets = train_y[i].to(self.device)
                
                h = tuple([each.data for each in h])
                
                self.model.zero_grad()
                output, h = self.model(inputs, h)
                
                loss = self.loss_fn(output,targets.contiguous().view(-1))
                
                running_loss += loss/batch_size
                
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(),self.clip)
                self.optimizer.step()
            
            perplexity =  torch.exp(running_loss)
            loss_list.append(running_loss.item())
            perp_list.append(perplexity.item())
            
            print("Epoch: {0} | Loss: {1}| Perplexity: {2}".format(epoch+1,loss_list[-1], perp_list[-1]))
            
            if perp_list[-1]<1.03:
                print('Perplexity below 1.03')
                count_perp +=1
        
            if perp_list[-1]>1.03:
                count_perp = 0

            if count_perp == 3:
                print('Perplexity below 1.03 for more then 3 iterations -- STOPPING ITERATIONS')
                break
        
        print('SAVING MODEL')
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': running_loss,
                    'loss_list': loss_list,
                    'perp_list':perp_list,
                    }, model_name)
        return loss_list, perp_list
    

def predict(net,tokenizer,letter,h,sampling):
    
    token = tokenizer.vocab.string_to_id[letter]
    
    x= np.array([[token]])

    input = torch.from_numpy(x)

    if torch.cuda.is_available():
        input = input.cuda()
    
    out, h = net(input, h)
    
    p = F.softmax(out, dim=1).data
    p = p.cpu()

    if sampling: 
        result = torch.multinomial(p, num_samples=1)
        return tokenizer.vocab.id_to_string[int(result)], h
    
    else: 
        _, idx = torch.topk(p,k=1,dim=-1)
        return tokenizer.vocab.id_to_string[int(idx)], h  
    
def generate(net,tokenizer,size,beginning='it',sampling=False):
    
    net.eval()
    if torch.cuda.is_available():
        h = (torch.zeros(net.num_layers, 1, net.lstm_size).cuda(),
             torch.zeros(net.num_layers, 1, net.lstm_size).cuda())
    else:
        h = (torch.zeros(net.num_layers, 1, net.lstm_size),
             torch.zeros(net.num_layers, 1, net.lstm_size))
    
    letters = []
    ids = []

    for i in beginning:
        letters.append(i)

    for i in range(size-len(beginning)):
        letter_out, h = predict(net,
                                tokenizer = tokenizer,
                                letter=letters[-1],
                                h=h,
                                sampling=sampling)
        letters.append(letter_out)
    
    return letters