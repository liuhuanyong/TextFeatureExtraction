# TextFeatureExtraction
Self complement of text feature extraction using algorithms including CHI, DF, IG, MI for the experiment of text classification based on sougou online news
基于卡方检验CHI，文档频率DF, 信息增益IG，互信息MI的文本特征提取与实现

# 引入
from feature_extract import *
dataer = FeatureExtract()  
# 设定提取特征数目，设置为5000  
features_num = 5000  
# 基于词语文档频率的特征词提取  
features = dataer.DF(feature_num)  
# 基于词语卡方信息的特征词提取  
features = dataer.CHI(feature_num)  
# 基于词语互信息的特征词提取  
features = dataer.MI(feature_num)  
# 基于词语信息增益频率的特征词提取   
features = dataer.IG(feature_num)  

# 输入：
data/data.txt: 搜狗文本分类语料库，共10个类别：  
'0': '汽车',  
'1': '财经',  
'2': 'IT',  
'3': '健康',  
'4': '体育',  
 '5': '旅游',  
'6': '教育',  
'7': '招聘',  
'9': '军事',  
 data.txt格式: category_id, word1 word2 word3 ...... wordn    
 # 输出:
 相应特征提取算法输出的文本特征，详细见：  
 data/features/chi.txt --> 卡方信息算法得到的文本特征TOP5000  
 data/features/df.txt --> 文档频率算法得到的文本特征TOP5000  
 data/features/mi.txt --> 互信息算法得到的文本特征TOP5000  
 data/features/ig.txt --> 信息增益算法得到的文本特征TOP5000  
 # 举例top20：
CHI: 训练，gt，一汽大众，都被，cnnic，中层，痛经，java，海岛，疲乏，区间，传送，领导能力，胜任，总社，尿液，诸侯，轻度，死亡，出汗  
DF：中国，公司，记者，到了，市场，时间，发展，这是，包括，工作，提供，都是，汽车，一种，国家，选择，情况，这一，北京，出了  
MI：中旅，蒙牛，总后勤部，60架，起飞时间，夏代，臣子，铬铁，末年，amd，卧槽，首回合，普吉，定位球，经济困难，忙忙碌碌，德智体，湖人，就业网，高血压  
IG：汽车，车型，轿车，找到，比赛，一页，发动机，消费者，品牌，www.sogou.com，搜狗，下一，上市，市场，旅游，销售，考生，公司，编辑，搜索   
 
