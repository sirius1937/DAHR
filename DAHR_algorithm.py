import random
import torch
import numpy as np
# import DAHR_algoritm

from transformers import ElectraTokenizer, ElectraModel 
import numpy 
import random
from sklearn.cluster import KMeans


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

    closest_to_all_index = np.argmin(result)

    distances_to_closest = np.array([np.linalg.norm(all_embedding_np[closest_to_all_index] - all_embedding_np[j]) for j in range(le)])
    
    distances_to_closest[closest_to_all_index] = np.inf
    
    closest_k_minus_1_indexes = np.argsort(distances_to_closest)[:k-1]
    
    return np.append([closest_to_all_index], closest_k_minus_1_indexes)






def sample(questions, all_embedding_np, pc=0.02, cluster_se="rd", branch=2, cluster_size=100, depth=0):
    ret, dep = [], []
    if len(questions) == 0:
        return ret, dep
    if len(questions) < cluster_size:
        # count = max(1, min(int(len(questions) * pc), len(questions)))
        count = max(1, min(int(len(questions) * pc) + 1, len(questions)))
        
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
    # iddd = numpy.array(random.sample([i for i in range(len(sample_results))],real_count))

    iddd = np.argsort(sample_results_depths)[-real_count:] 

    r_questions, r_depths = sample_results[iddd], sample_results_depths[iddd]

    indexs = []
    ret_dataset = []
    for i in r_questions:
        indexs.append(questions.index(i))
    for i in indexs:
        ret_dataset.append(datasets[i])
    return indexs,r_questions, r_depths, ret_dataset
