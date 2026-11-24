### Drug-Drug Interaction (DDI) Prediction Exercise

############ Project 1 ############ 

#train_file="/home/ioanna/metaptixiako/kanterakis/projects/DDI-Bench-main/DDI_Ben/DDI_Ben/data/drugbank_random/train.txt"
#test_file="/home/ioanna/metaptixiako/kanterakis/projects/DDI-Bench-main/DDI_Ben/DDI_Ben/data/drugbank_random/test_S0.txt"
#relation_json="/home/ioanna/metaptixiako/kanterakis/projects/DDI-Bench-main/DDI_Ben/TextDDI/data/drugbank_random/relation2id.json"
#molecular_feats="/home/ioanna/metaptixiako/kanterakis/projects/DDI-Bench-main/DDI_Ben/DDI_Ben/data/initial/drugbank/DB_molecular_feats.pkl"

import argparse
import pickle
import json
import numpy as np

def load_pickle_data(filename): #load molecular_feats file (.pkl) and returns dictionaries
    with open(filename, 'rb') as f:
        data=pickle.load(f)
    drugbank_ids=data['DrugBank ID']
    node_ids=data['Node ID']
    smiles=data['SMILES']
    morgan_feats=data['Morgan_Features']
    rdkit_feats=data['RDKit2D_Features']
    
    id_to_feats={} #match node id to features (morgan and rdkit)
    db_to_nid={} #match drugbank id to node id

    for i in range(len(node_ids)):
        nid=node_ids[i]
        dbid=drugbank_ids[i]
        
        morgan_features=np.array(morgan_feats[i],dtype=np.float32)
        morgan_features=np.where(morgan_features>0,1.0,0.0)
        rdkit_features=np.array(rdkit_feats[i],dtype=np.float32)

        id_to_feats[nid]={'morgan':morgan_features,'rdkit':rdkit_features} #we save them as array so that we can ddo math operations later
        db_to_nid[dbid]=nid
    
    return id_to_feats, db_to_nid


def calculate_tanimoto_distance(v1,v2): #calculates tanimoto distance between two binary vectors (for morgan features)
    numerator=np.dot(v1,v2) #the numerator is the intersection of the 2 vectors (we can use the dot product because we have binary vectors)
    denominator=np.sum(v1)+np.sum(v2)-numerator #the denominator is the union of the 2 vectors
    if denominator==0: #if the vectors are empty
        if np.sum(v1)==np.sum(v2):
            return 0.0 #if the vectors are completely similar, the distance is 0
        else:
            return 1.0 #if the vectors are completely different, the distance is 1
        
    similarity=numerator/denominator
    return 1.0-similarity #the distance between the 2 vectors


def calculate_cosine_distance(v1,v2): #calculates the cosine distance between 2 vectors (non binary) (for rdkit features)
    numerator=np.dot(v1,v2) #the numerator is the dot product of the 2 vectors
    len_v1=np.linalg.norm(v1) #the length of the vector
    len_v2=np.linalg.norm(v2) #the length of the vector
    if len_v1==0 or len_v2==0:
        return 1.0 #if a vector is 0, the distance is 1
    denominator=len_v1*len_v2 #the denominator is the product of the vectors' lengths
    similarity=numerator/denominator
    similarity=np.clip(similarity,-1.0,1.0) #we roundup the similarity so that it is between -1 and 1
    return 1.0-similarity #the distance between the 2 vectors


def calculate_pair_distance(feats_test_pair, feats_train_pair, mf_weight=0.5, rd_weight=0.5):
    test1, test2=feats_test_pair #the 2 test drugs
    train1,train2=feats_train_pair #the 2 train drugs
    
    #Straight comparison
    tanimoto_1=calculate_tanimoto_distance(test1['morgan'], train1['morgan']) #we compare the 1st drug from test and train sets
    tanimoto_2=calculate_tanimoto_distance(test2['morgan'], train2['morgan']) #we compare the 2nd drug from test and train sets
    mf_distance=(tanimoto_1+tanimoto_2)/2.0 #the average tanimoto distance

    cosine_1=calculate_cosine_distance(test1['rdkit'], train1['rdkit']) #we compare the 1st drug from test and train sets
    cosine_2=calculate_cosine_distance(test2['rdkit'], train2['rdkit']) #we compare the 2nd drug from test and train sets
    rd_distance=(cosine_1+cosine_2)/2.0 #the average cosine distance
    
    total_distance=(mf_weight*mf_distance)+(rd_weight*rd_distance)

    return total_distance #the distance betweet the 2 pairs

parser=argparse.ArgumentParser()
parser.add_argument('--molecular_feats',required=True,help='Path to the molecular features pickle file')
parser.add_argument('--relation2id',help='Path to the relation ID mapping JSON file')
parser.add_argument('--train',required=True,help='Path to the training data file')
parser.add_argument('--mode',required=True, choices=['train', 'inference'],help='Operation mode, must be either train or inference')
parser.add_argument('--test',help='Path to the test data file. Required when --mode train, optional for inference mode')
parser.add_argument('--cutoff', type=int, default=None,help='Integer specifying the maximum number of test samples to process. If not specified, all test samples will be processed')
parser.add_argument('--drugbank_1',help='First DrugBank ID. Required when --mode inference')
parser.add_argument('--drugbank_2',help='Second DrugBank ID. Required when --mode inference')
args=parser.parse_args()


id_to_features, db_to_node_id= load_pickle_data(args.molecular_feats) #load features

train_data_pairs=[]
with open(args.train,'r') as f: #load training data
    for line in f:
        line=line.strip().split()
        train_data_pairs.append((int(line[0]), int(line[1]), int(line[2])))


if args.mode=='train':
    if args.test:
        test_data_pairs=[]
        with open(args.test,'r') as f: #load testing data
            for line in f:
                line=line.strip().split()
                test_data_pairs.append((int(line[0]), int(line[1]), int(line[2])))
        
        if args.cutoff:
            test_data_pairs_final=test_data_pairs[:args.cutoff] #we keep only the first 'cutoff' samples

        number_of_correct_predictions=0
        total_number_of_test_samples=0

        for index, (test_drug1,test_drug2,test_relation) in enumerate(test_data_pairs_final): #for each pair in the test set
            if test_drug1 not in id_to_features or test_drug2 not in id_to_features: #we check if the drug are in the features database
                continue
            test_pair_features=(id_to_features[test_drug1], id_to_features[test_drug2]) #we get the features of the 2 test drugs
            
            min_distance=float('inf')
            predicted_relation=-100

            for train_drug1, train_drug2, train_relation in train_data_pairs: #we compare with all the pairs in the train set
                if train_drug1 in id_to_features and train_drug2 in id_to_features:
                    train_pair_features=(id_to_features[train_drug1], id_to_features[train_drug2]) #we get the features of the 2 train drugs

                    pair_distance=calculate_pair_distance(test_pair_features,train_pair_features) #we calculate the pair distance
                    if pair_distance<min_distance: #if we find a closer sample, we save it
                        min_distance=pair_distance
                        predicted_relation=train_relation
                        if min_distance==0: #if the distance is 0 (the same drugs) we stop beacuse we will not find something better
                            break
            
            if predicted_relation==test_relation: #we check if the prediction is correct
                number_of_correct_predictions+=1
            total_number_of_test_samples+=1

            print(f'Sample: {index+1}/{len(test_data_pairs)}, Predicted: {predicted_relation}, Real: {test_relation}')

        if total_number_of_test_samples>0: #we calculate the accuracy
            accuracy=number_of_correct_predictions/total_number_of_test_samples
        else:
            accuracy=0

        print(f"Total: {total_number_of_test_samples}, Correct: {number_of_correct_predictions}, Wrong: {total_number_of_test_samples-number_of_correct_predictions}")
        print(f"Accuracy: {accuracy}")

elif args.mode=='inference':
    if args.relation2id and args.drugbank_1 and args.drugbank_2:
        db1=args.drugbank_1
        db2=args.drugbank_2

        if db1 in db_to_node_id and db2 in db_to_node_id: #we get the node id of each drugbase id
            drug1_id=db_to_node_id[db1]
            drug2_id=db_to_node_id[db2]

            target_pair_features=(id_to_features[drug1_id], id_to_features[drug2_id]) #we get the features of the 2 target drugs

            min_distance=float('inf')
            predicted_relation=-100

            for train_drug1, train_drug2, train_relation in train_data_pairs: #we compare with all the pairs in the train set
                if train_drug1 in id_to_features and train_drug2 in id_to_features:
                    train_pair_features=(id_to_features[train_drug1], id_to_features[train_drug2]) #we get the features of the 2 train drugs

                    pair_distance=calculate_pair_distance(target_pair_features,train_pair_features) #we calculate the pair distance
                    if pair_distance<min_distance: #if we find a closer sample, we save it
                        min_distance=pair_distance
                        predicted_relation=train_relation
                        if min_distance==0: #if the distance is 0 (the same drugs) we stop beacuse we will not find something better
                            break
            
            with open(args.relation2id,'r') as f:
                relation_data=json.load(f)
            
            description=relation_data.get(str(predicted_relation), 'Unknown') #we get the prediction of the interaction
            print(f'Prediction: {description}')