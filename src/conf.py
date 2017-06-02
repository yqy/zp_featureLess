import argparse

#parse arguments
parser = argparse.ArgumentParser(description="Experiemts for ZP resolution (by qyyin)\n")
parser.add_argument("-data",default="None",help="specify data file")
parser.add_argument("-type",default="None",help="azp:Get azp feature ***\n\r res:Get res feature")
parser.add_argument("-res_t",default="0.5",help="Threshold for resolution classification")
parser.add_argument("-azp_t",default="0.5",help="Threshold for AZP classification")
parser.add_argument("-res_pos",default="1",help="Postive instance for resolution classification")
parser.add_argument("-embedding",default="/Users/yqy/work/data/word2vec/embedding.ontonotes",help="embedding dir")
parser.add_argument("-embedding_dimention",default=100,type=int,help="embedding dimention")
parser.add_argument("-test_data",default="None",help="Test data for DeepLearning")

parser.add_argument("-echos",default=10,type=int,help="Echo Times")
parser.add_argument("-lr",default=0.03,type=float,help="Learning Rate")
parser.add_argument("-batch",default=15,type=int,help="batch size")
parser.add_argument("-dev_prob",default=0.1,type=float,help="probability of development set")
parser.add_argument("-dropout_prob",default=0.5,type=float,help="probability of dropout")

args = parser.parse_args()

'''
type:
# azp : get azp features
# res : get res features 
# gold : get result with  --  gold AZP + gold Parse
# auto : get result with  --  auto AZP + gold Parse
# system : get result with -- auto AZP + auto Parse
# nn : train for nerual network 
'''
