path = "/home/caozx/projects/DAHR/dataset_sq/"


from utils import get_dataset,BuboDataset

train,dev,test = get_dataset(path)
# # 
print(len(train))
print(len(dev))
print(len(test))


print(train[0].__dict__)
print(dev[0].__dict__)
print(test[0].__dict__)


print("type_train = {}".format(type(train)))
train = BuboDataset(train)

print("type_train = {}".format(type(train)))

# print(tmp.__dict__)

# '''
# {'sub': ['m', '04whkz5'], 'rela': ['book', 'written_work', 'subjects'], 'obj': ['m', '01cj3p'], 'text': ['what', 'is', 'the', 'book', 'e', 'about']}
# '''

