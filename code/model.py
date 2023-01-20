import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Dropout,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from layers.graph import GraphLayer,GraphConv


class KerasMultiSourceDualGCNModel(object):
    def __init__(self,use_gexpr=True,use_cn=True,regr=True):#
        self.use_gexpr = use_gexpr
        self.use_cn = use_cn
        self.regr = regr

    def createMaster(self, drug_dim, cell_line_dim, 
                    drug_GCN_units_list=[256, 128], 
                    cell_feature_fc_units_list=[32, 128],
                    fc_units_list=[256, 128, 10],
                    cell_line_gcn_units_list=[256, 256, 256, 256],
                    universal_dropout=0.1,
                    fc_layers_dropout=[0.3, 0.2, 0]):
        # drug-graph input layer
        drug_feat_input = Input(shape=(None,drug_dim),name='drug_feat_input')
        drug_adj_input = Input(shape=(None,None),name='drug_adj_input')
        
        # bio-graph input layer
        cell_line_feat_input = Input(shape=(None,cell_line_dim),name='cell_line_feat_input')
        cell_line_adj_input = Input(shape=(None,None),name='cell_line_adj_input')
        
        # drug-GCN
        GCN_layer = GraphConv(units=drug_GCN_units_list[0],step_num=1,name="DrugGraph_1_GCN")([drug_feat_input,drug_adj_input])
        GCN_layer = Activation('relu')(GCN_layer)
        GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(universal_dropout, name="DrugGraph_1_out")(GCN_layer)
        
        GCN_layer = GraphConv(units=drug_GCN_units_list[1],step_num=1,name="DrugGraph_last_GCN")([GCN_layer,drug_adj_input])
        GCN_layer = Activation('relu')(GCN_layer)
        GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(universal_dropout, name="DrugGraph_last_out")(GCN_layer)

        x_drug = GlobalAveragePooling1D(name="DrugGraph_out")(GCN_layer)

        # bio-graph GCN
        cell_line_feat_input_high = Dense(cell_feature_fc_units_list[0], activation = 'tanh')(cell_line_feat_input) 
        cell_line_feat_input_high = Dropout(universal_dropout)(cell_line_feat_input_high)
        cell_line_feat_input_high = Dense(cell_feature_fc_units_list[1], activation = 'tanh')(cell_line_feat_input_high) 
        cell_line_feat_input_high = Dropout(universal_dropout)(cell_line_feat_input_high) 

        cell_line_GCN = GraphConv(units=cell_line_gcn_units_list[0],step_num=1,name="CelllineGraph_1_GCN")([cell_line_feat_input_high,cell_line_adj_input])
        cell_line_GCN = Activation('relu')(cell_line_GCN)
        cell_line_GCN = BatchNormalization()(cell_line_GCN)
        cell_line_GCN = Dropout(universal_dropout)(cell_line_GCN)

        if len(cell_line_gcn_units_list) > 1:
            cell_line_GCN = GraphConv(units=cell_line_gcn_units_list[1],step_num=1)([cell_line_GCN,cell_line_adj_input])
            cell_line_GCN = Activation('relu')(cell_line_GCN)
            cell_line_GCN = BatchNormalization()(cell_line_GCN)
            cell_line_GCN = Dropout(universal_dropout)(cell_line_GCN)

        if len(cell_line_gcn_units_list) > 2:
            cell_line_GCN = GraphConv(units=cell_line_gcn_units_list[2],step_num=1)([cell_line_GCN,cell_line_adj_input])
            cell_line_GCN = Activation('relu')(cell_line_GCN)
            cell_line_GCN = BatchNormalization()(cell_line_GCN)
            cell_line_GCN = Dropout(universal_dropout)(cell_line_GCN)

        if len(cell_line_gcn_units_list) > 3:
            cell_line_GCN = GraphConv(units=cell_line_gcn_units_list[3],step_num=1)([cell_line_GCN,cell_line_adj_input])
            cell_line_GCN = Activation('relu')(cell_line_GCN)
            cell_line_GCN = BatchNormalization()(cell_line_GCN)
            cell_line_GCN = Dropout(universal_dropout)(cell_line_GCN)

        x_cell_line = GlobalAveragePooling1D(name="CelllineGraph_out")(cell_line_GCN)

        x = Concatenate(name="Merge_Drug_Cellline_graphs")([x_cell_line,x_drug])
        x = Dense(fc_units_list[0], activation = 'tanh')(x)
        x = Dropout(fc_layers_dropout[0])(x)
        x = Dense(fc_units_list[1], activation = 'tanh')(x)
        x = Dropout(fc_layers_dropout[1])(x)
        x = Dense(fc_units_list[2], activation = 'tanh')(x)
        x = Dropout(fc_layers_dropout[2])(x)
        output = Dense(1, name='output')(x)

        model  = Model(inputs=[drug_feat_input,drug_adj_input,cell_line_feat_input,cell_line_adj_input],outputs=output)  

        return model
