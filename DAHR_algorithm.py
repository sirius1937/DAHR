import random
import torch
import numpy as np
# import DAHR_algoritm

from transformers import ElectraTokenizer, ElectraModel 
import numpy 
import random
from sklearn.cluster import KMeans

# random.seed(42)

# def min_pos(all_embedding_np,k):
#     le = len(all_embedding_np)
#     result = np.array([0]*le,dtype=float)
#     for i in range(le):
#         for j in range(i):
#             r = np.linalg.norm(all_embedding_np[i] - all_embedding_np[j])
#             result[i] += r
#             result[j] += r
#     return np.argsort(result)[:k] ## min_pos

# def min_pos(all_embedding_np, k):
#     all_embedding_tensor = torch.tensor(all_embedding_np).cuda()
#     dist_matrix = torch.norm(all_embedding_tensor[:, None] - all_embedding_tensor, dim=2, p=2)
#     result = torch.sum(dist_matrix, dim=1)
#     _, indices = torch.topk(result, k, largest=False)
#     return indices.cpu().numpy() 






import numpy as np
import random
from sklearn.cluster import KMeans


import numpy as np

def min_pos(all_embedding_np, k):
    le = len(all_embedding_np)
    result = np.array([0]*le, dtype=float)
    for i in range(le):
        for j in range(i):
            r = np.linalg.norm(all_embedding_np[i] - all_embedding_np[j])
            result[i] += r
            result[j] += r
    # 找到离大家最近的那一个embedding的索引
    closest_to_all_index = np.argmin(result)
    # 计算这个embedding到所有其他embedding的距离
    distances_to_closest = np.array([np.linalg.norm(all_embedding_np[closest_to_all_index] - all_embedding_np[j]) for j in range(le)])
    # 将离大家最近的那一个embedding的距离设置为无穷大，以便在接下来的操作中排除它
    distances_to_closest[closest_to_all_index] = np.inf
    # 找到离这个最近的k-1个其他embedding的索引
    closest_k_minus_1_indexes = np.argsort(distances_to_closest)[:k-1]
    # 返回离大家最近的那一个，以及离这个最近的k-1个其他embedding的索引
    return np.append([closest_to_all_index], closest_k_minus_1_indexes)




# def min_pos(all_embedding_np, k):
#     # 这里假设min_pos函数的实现与之前讨论的相同
#     ...

def sample(questions, all_embedding_np, pc=0.02, cluster_se="rd", branch=2, cluster_size=100, depth=0):
    ret, dep = [], []
    if len(questions) == 0:
        return ret, dep
    if len(questions) < cluster_size:
        count = max(1, min(int(len(questions) * pc), len(questions)))
        # count = max(1, min(int(len(questions) * pc) + 1, len(questions)))
        iddd = np.array(random.sample(range(len(questions)), count))
        if cluster_se == "md":
            iddd = min_pos(all_embedding_np, count)
        ret = questions[iddd]
        dep = [depth + 1] * len(ret)
        return ret, dep

    # 使用KMeans进行聚类
    y_pred = KMeans(n_clusters=branch, random_state=9).fit_predict(all_embedding_np)
    for i in range(branch):
        id_i = np.where(y_pred == i)
        question_i = questions[id_i].copy()
        emb_i = all_embedding_np[id_i].copy()
        p, q = sample(question_i, emb_i, pc=pc, cluster_se=cluster_se, branch=branch, cluster_size=cluster_size, depth=depth+1)
        ret.append(p)
        dep.append(q)
    
    # 将结果连接并返回
    return np.concatenate(ret), np.concatenate(dep)

# 注意：这里的代码仅为示例，具体实现可能需要根据实际情况进行调整。
# 例如，min_pos函数需要根据前文的讨论来实现。




# def sample(questions, all_embedding_np,pc=0.02,cluster_se="rd",branch=2,cluster_size=100,depth=0):
#     ret , dep = [], []
#     if len(questions) == 0:
#         return ret,dep
#     if len(questions) < int(cluster_size) :
#         count = int(len(questions)*(pc)) + 1
#         count = min(count, len(questions))
#         count = max(1,count)
#         iddd = numpy.array(random.sample([i for i in range(len(questions))],count))
#         if cluster_se == "md":
#             iddd = min_pos(all_embedding_np,count)
#         ret = questions[iddd]
#         dep = depth + 1
#         return ret, [dep] * len(ret)

#     if branch == 2:
#         y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(all_embedding_np)
#         id_0 = numpy.where(y_pred==0)
#         id_1 = numpy.where(y_pred==1)
#         question_0 = questions[id_0].copy()
#         question_1 = questions[id_1].copy()
#         emb_0 = all_embedding_np[id_0].copy()
#         emb_1 = all_embedding_np[id_1].copy()
#         p,q = sample(question_0, emb_0, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         r,s = sample(question_1, emb_1, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         return numpy.concatenate((p, r)),numpy.concatenate((q, s))

#     elif branch == 3:
#         y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(all_embedding_np)
#         id_0 = numpy.where(y_pred==0)
#         id_1 = numpy.where(y_pred==1)
#         id_2 = numpy.where(y_pred==2)
#         question_0 = questions[id_0].copy()
#         question_1 = questions[id_1].copy()
#         question_2 = questions[id_2].copy()
#         emb_0 = all_embedding_np[id_0].copy()
#         emb_1 = all_embedding_np[id_1].copy()
#         emb_2 = all_embedding_np[id_2].copy()
#         p,q = sample(question_0, emb_0, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         r,s = sample(question_1, emb_1, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         rr,ss = sample(question_2, emb_2, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         return numpy.concatenate((p, r, rr)),numpy.concatenate((q, s, ss))
    
#     elif branch == 4:
#         y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(all_embedding_np)
#         id_0 = numpy.where(y_pred==0)
#         id_1 = numpy.where(y_pred==1)
#         id_2 = numpy.where(y_pred==2)
#         id_3 = numpy.where(y_pred==2)
#         question_0 = questions[id_0].copy()
#         question_1 = questions[id_1].copy()
#         question_2 = questions[id_2].copy()
#         question_3 = questions[id_3].copy()
#         emb_0 = all_embedding_np[id_0].copy()
#         emb_1 = all_embedding_np[id_1].copy()
#         emb_2 = all_embedding_np[id_2].copy()
#         emb_3 = all_embedding_np[id_3].copy()
#         p,q = sample(question_0, emb_0, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         r,s = sample(question_1, emb_1, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         t,u = sample(question_2, emb_2, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         v,w = sample(question_3, emb_3, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         return numpy.concatenate((p, r, t, v)),numpy.concatenate((q, s, u, w))  
#     elif branch == 5:
#         y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(all_embedding_np)
#         id_0 = numpy.where(y_pred==0)
#         id_1 = numpy.where(y_pred==1)
#         id_2 = numpy.where(y_pred==2)
#         id_3 = numpy.where(y_pred==3)
#         id_4 = numpy.where(y_pred==4)
#         question_0 = questions[id_0].copy()
#         question_1 = questions[id_1].copy()
#         question_2 = questions[id_2].copy()
#         question_3 = questions[id_3].copy()
#         question_4 = questions[id_4].copy()
#         emb_0 = all_embedding_np[id_0].copy()
#         emb_1 = all_embedding_np[id_1].copy()
#         emb_2 = all_embedding_np[id_2].copy()
#         emb_3 = all_embedding_np[id_3].copy()
#         emb_4 = all_embedding_np[id_4].copy()
#         p,q = sample(question_0, emb_0, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         r,s = sample(question_1, emb_1, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         t,u = sample(question_2, emb_2, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         v,w = sample(question_3, emb_3, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         x,y = sample(question_4, emb_4, pc = pc,cluster_se=cluster_se,branch=branch,cluster_size=cluster_size,depth=depth+1)
#         return numpy.concatenate((p, r, t, v, x)),numpy.concatenate((q, s, u, w, y))        


    # 
    # return numpy.concatenate((p, r, rr)),numpy.concatenate((q, s, ss))
    # y_pred = KMeans(n_clusters=5, random_state=9).fit_predict(all_embedding_np)
    # id_0 = numpy.where(y_pred==0)
    # id_1 = numpy.where(y_pred==1)
    # id_2 = numpy.where(y_pred==2)
    # id_3 = numpy.where(y_pred==3)
    # id_4 = numpy.where(y_pred==4)

    # question_0 = questions[id_0].copy()
    # question_1 = questions[id_1].copy()
    # question_2 = questions[id_2].copy()
    # question_3 = questions[id_3].copy()
    # question_4 = questions[id_4].copy()

    # emb_0 = all_embedding_np[id_0].copy()
    # emb_1 = all_embedding_np[id_1].copy()
    # emb_2 = all_embedding_np[id_2].copy()
    # emb_3 = all_embedding_np[id_3].copy()
    # emb_4 = all_embedding_np[id_4].copy()

    # p,q = sample(question_0, emb_0, depth=depth+1, pc=pc)
    # r,s = sample(question_1, emb_1, depth=depth+1, pc=pc)
    # rr,ss = sample(question_2, emb_2, depth=depth+1, pc=pc)
    # t,u = sample(question_3, emb_3, depth=depth+1, pc=pc)
    # v,w = sample(question_4, emb_4, depth=depth+1, pc=pc)
    # return numpy.concatenate((p, r, rr, t, v)), numpy.concatenate((q, s, ss, u, w))



def sample_function(datasets, percent = 0.02, cluster_se = "rd",branch = 2, cluster_size = 150):

    questions = [i.raw_text for i in datasets]
    def q2embedding(sentences):
        model_name = "./electra-small"
        tokenizer = ElectraTokenizer.from_pretrained(model_name)
        model = ElectraModel.from_pretrained(model_name)
        batch_size = 64  # 可以根据需要调整批次大小
        batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        all_embeddings = []
        for batch in batches:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings)
        all_embeddings_np = all_embeddings.numpy()
        all_embeddings_np = all_embeddings_np.astype(numpy.float64)
        return all_embeddings_np
    embeddings = q2embedding(questions)
    r_questions, r_depths = sample(numpy.array(questions),embeddings, cluster_se=cluster_se,branch=branch,cluster_size = cluster_size, pc=percent)
    real_count = int ( len(questions) * percent )
    # if real_count < len(r_questions):
    #     iddd = numpy.array(random.sample([i for i in range(len(r_questions))],real_count))
    #     r_questions, r_depths = r_questions[iddd], r_depths[iddd]
    # if real_count > len(r_questions):
    #     k = 
    #     app = numpy.array(random.sample([i for i in range(len(r_questions))],real_count - len(r_questions)))

    sample_results = []
    sample_results_depths = []
    while len(sample_results) < real_count :
        sample_results = numpy.concatenate((sample_results, r_questions))
        sample_results_depths = numpy.concatenate((sample_results_depths, r_depths))
    iddd = numpy.array(random.sample([i for i in range(len(sample_results))],real_count))
    r_questions, r_depths = sample_results[iddd], sample_results_depths[iddd]

    indexs = []
    ret_dataset = []
    for i in r_questions:
        indexs.append(questions.index(i))
    for i in indexs:
        ret_dataset.append(datasets[i])
    return indexs,r_questions, r_depths, ret_dataset
