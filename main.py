import torch
import numpy as np
import random
import os

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # 如果使用多个GPU
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



from argparse import ArgumentParser

import random
import log
import os,torch
from transformers import ElectraConfig,ElectraTokenizer
from modeling_baseline import ElectraForMultipleChoicePlus
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW,get_cosine_schedule_with_warmup
import numpy as np
import DAHR_algorithm
from calc_metric import calc_metric


from utils import get_dataset, BuboDataset
from utils import prompt
from utils import _split,padding,generate_batch,calc_acc


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def data_update(hy_args,train,prompt_method="none"):

    '''
    对于数据集的更新：
        每个数据本身的data:
            rel     
    '''
    train = sorted(train,key=lambda x:x.rela)     
    for t in train:
        t.label = 0
    LEN = len(train)
    for idx,t in enumerate(train) :
        idx_li = []
        if idx < 500 :
            idx_li = [i for i in range(idx+1,idx+hy_args.choice_cnt)]
        else :
            idx_li = [i for i in range(idx-hy_args.choice_cnt+1,idx)]

        while True:
            idx_li = random.sample(range(0,LEN),hy_args.choice_cnt-1)
            if idx_li.count(idx) == 0:
                break
        t.choice = []
        t.choice.append(padding(hy_args.tokenizer,prompt(t.text,t.sub,t.rela,method=prompt_method),hy_args.case_length))

        # print(t.text)
        # print(t.sub)
        # print(t.rela)
        # import time
        for i in idx_li :
            if i%5 != 0 :
                t.choice.append(padding(hy_args.tokenizer,prompt(t.text,t.sub,train[i].rela,method=prompt_method),hy_args.case_length)) ## 
            else :
                t.choice.append(padding(hy_args.tokenizer,prompt(t.text,train[i].sub,t.rela,method=prompt_method),hy_args.case_length))
        # t.choice[-1] = padding(hy_args.tokenizer,prompt(t.text,t.rela,t.sub,method=prompt_method),hy_args.case_length)
        sw_idx = random.sample(range(0,hy_args.choice_cnt),1)[0]
        tmp = t.choice[sw_idx]
        t.choice[sw_idx] = t.choice[0]
        t.choice[0] = tmp
        t.label = sw_idx
    for t in train:
        t.raw_text = " ".join(t.text)
        t.raw_sub = t.sub
        t.raw_rela = t.rela
        t.text = padding(hy_args.tokenizer, t.text, hy_args.case_length)
        t.sub = padding(hy_args.tokenizer, t.sub, hy_args.case_length)
        t.rela = padding(hy_args.tokenizer, t.rela, hy_args.case_length)

    # random.shuffle(train)   ## 好像shuffle了不太好。
    return train


def run(hy_args, Tasks, ta_logs):
    def _train():
        model.train()
        for epoid in range(hy_args.train_epochs):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step,batch in enumerate(train_iter):
                question, choices, label = batch
                choices.to(device)
                label.to(device)
                # print(choices.shape)
                # print(label.shape)
                # print(question.shape)
                outputs = model(**{
                    "input_ids" : choices,
                    "labels" : label
                })
                loss = outputs[0]
                loss = loss.mean()
                logits = outputs[1]
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += label.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
    def _test() :
        eval_loss = 0
        nb_eval_steps = 0
        preds = None
        model.eval()
        for step,batch in enumerate(test_iter):
            # a,b,c,d,e = batch
            # a = a.to(device)
            # d = d.to(device)
            # e = e.to(device)
            question, choices, label = batch
            choices.to(device)
            label.to(device)
            # print(choices.shape)
            # print(label.shape)
            with torch.no_grad():
                outputs = model(**{
                    "input_ids" : choices,
                    "labels" : label
                })
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.detach().mean().item()
            
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, label.detach().cpu().numpy(), axis=0)
        acc = calc_acc(preds,out_label_ids)
        return acc
    

    print("使用显卡为{}".format(hy_args.use_nvidia))

    ###  测试！！！
    # os.environ["CUDA_VISIBLE_DEVICES"] = hy_args.use_nvidia
    model_name = hy_args.model_name
    config = ElectraConfig.from_pretrained(model_name,
                                                num_labels=hy_args.choice_cnt,
                                                )
    tokenizer = ElectraTokenizer.from_pretrained(model_name,
                                                    )
    model = ElectraForMultipleChoicePlus.from_pretrained(model_name,
                                            config=config,
                                            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=hy_args.learning_rate, weight_decay=0.0)
    
    all_round_score = []
    import numpy as np
    detail_matrix = np.zeros((5, 5))

    for i in range(5):
        train_iter = DataLoader(Tasks[i], batch_size=hy_args.batch_size,  shuffle=True,collate_fn=generate_batch)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=len(train_iter) * hy_args.batch_size * 0.05, num_training_steps=len(train_iter)* hy_args.batch_size
        )
        _train()
        sum = 0
        for j in range(i+1) :
            test_iter = DataLoader(Tasks[j], batch_size=hy_args.batch_size,  shuffle=True,collate_fn=generate_batch)
            acc = _test()
            sum += acc
            ta_logs.info("i= {} ,j= {} ,acc= {} ".format(i,j,acc))
            print("i = ",i," j = ",j," acc = ",acc)
            detail_matrix[i][j] = acc
        print("===score :" , sum/(i+1))
        ta_logs.info("score = {}".format(sum/(i+1)))
        all_round_score.append(sum/(i+1))
    
    ta_logs.info("all_score = {}".format(all_round_score))
    metrics = calc_metric(detail_matrix, all_round_score)
    ta_logs.info("metric = {}".format(metrics))

    return detail_matrix, all_round_score, metrics[:3]





def solve(hy_args, Tasks, ta_logs):
    sp = hy_args.sample_percent
    if hy_args.replay_method == 'random':
        for i in range(1,5):
            Tasks[i] = Tasks[i] + random.sample(Tasks[i-1], int(len(Tasks[i-1])*sp) )
    elif hy_args.replay_method == 'DAHR_rd':
        for i in range(1,5):
            Tasks[i] = Tasks[i] + DAHR_algorithm.sample_function(Tasks[i-1],
                                                        percent=sp, cluster_se="rd",
                                                        branch=hy_args.branch,
                                                        cluster_size = hy_args.cluster_size
                                                                   )[-1]
    elif hy_args.replay_method == 'DAHR_md':
        for i in range(1,5):
            Tasks[i] = Tasks[i] + DAHR_algorithm.sample_function(Tasks[i-1],
                                                        percent=sp, 
                                                        cluster_se="md",
                                                        branch=hy_args.branch,
                                                        cluster_size = hy_args.cluster_size
                                                                   )[-1]
    elif hy_args.replay_method == "upper_bound":
        for i in range(1,5) :
            Tasks[i] = Tasks[i-1] + Tasks[i]
    elif hy_args.replay_method == "none":
        pass

    for i in range(5):
        Tasks[i] = BuboDataset(Tasks[i])
    for idx,item in enumerate(Tasks) :
        ta_logs.info("task = {}, size = {}".format(idx+1, len(item)))
        print("task = {}, size = {}".format(idx+1, len(item)))
    return run(hy_args, Tasks, ta_logs)
 

def main(hy_args, ta_logs):
    
    all_result = np.zeros((hy_args.default_times,3))
    for exp_i in range(hy_args.default_times):
        # random.seed(hy_args.random_seed)
        hy_args.tokenizer = ElectraTokenizer.from_pretrained(hy_args.model_name)
        train,dev,test = get_dataset()
        train = data_update(hy_args,train,prompt_method=hy_args.prompt_method)
        test = data_update(hy_args,test,prompt_method=hy_args.prompt_method)

        if hy_args.show_details:
            T = train[400].choice[0]
            T = hy_args.tokenizer.convert_ids_to_tokens(T)
            print(T)
            print(train[400].__dict__)
            ta_logs.info("train的类型是{}".format(type(train)))

        Tasks = []
        for i in range(5):
            Tasks.append(train[hy_args.task_cases * i :hy_args.task_cases * (i+1) ])
            random.shuffle(Tasks[i])
        if not hy_args.mini_test:
            Tasks = [train[:18001],train[18001:40056],train[40056:50824],train[50824:62496],train[62496:]]
        for i in range(5):
            random.shuffle(Tasks[i])
            print(len(Tasks[i]))

        ta_logs.info("\n\n{} th learning exp".format(exp_i))
        results = solve(hy_args, Tasks, ta_logs)
        print(results[2])
        all_result[exp_i,:] = results[2]
        ta_logs.info("result = {}".format(results))

    ta_logs.info(all_result)
    ta_logs.info(np.mean(all_result, axis=0))
    
if __name__ == '__main__':
    print("OK")
    parser = ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default="blank")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument('--use_nvidia',type=str,default='0-1')


    parser.add_argument('--case_length',type=int,default=50)
    parser.add_argument('--choice_cnt',type=int,default=40)

    parser.add_argument('--sample_percent',type=float,default=0.02)
    parser.add_argument('--train_epochs',type=int,default=3)
    parser.add_argument('--learning_rate',type=float,default=5e-5)
    parser.add_argument('--save_log_path',type=str,default='./default.log')

    parser.add_argument('--prompt_method',type=str,default='base')
    parser.add_argument('--replay_method',type=str,default='DAHR-md')

    parser.add_argument('--random_seed',type=int,default=1234)
    parser.add_argument('--default_times',type=int,default=1)

    parser.add_argument('--show_details',default=False,action='store_true')
    parser.add_argument('--branch',type=int,default=2)
    parser.add_argument('--cluster_size',type=int,default=100)

    parser.add_argument('--task_cases',type=int,default=15000)
    parser.add_argument('--mini_test',action='store_true',default=False)
    parser.add_argument('--model_name',type=str,default="./electra-small/")
    parser.add_argument('--desc',type=str,default="")


    hyperparams = parser.parse_args()


    set_seed( hyperparams.random_seed )


    hyperparams.use_nvidia = hyperparams.use_nvidia.replace('-',',')
    
    if hyperparams.mini_test:
        hyperparams.task_cases = 1000
        hyperparams.default_times = 3
 
 

    ta_logs = log.get_logger(hyperparams.save_log_path)   
    ta_logs.info("hyperparams: {} \n\n".format(hyperparams)) 
    ta_logs.info("experiment beginning")

    main(hyperparams, ta_logs)
    ta_logs.info("experiment ending")
 