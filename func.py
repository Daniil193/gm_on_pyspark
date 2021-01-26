from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql import Window
from graphviz import Digraph
import pandas as pd


class Graph_miner:
    
    def __init__(self, frame, payer, recipient, curr_sum, date, preprocessing=True):
        self.frame = frame.select([payer, recipient, curr_sum, date])
        self.payer = payer
        self.recipient = recipient
        self.curr_sum = curr_sum
        self.date = date
        self.result = None
        
        if preprocessing:
            self.__preprocess()
        
    def __get_time_stat(self):
        ww = Window.partitionBy([self.payer, self.recipient]).orderBy(self.date)
        lag_expr = f.lag(self.frame[self.date]).over(ww)
        self.frame = self.frame.withColumn("prev_time", lag_expr)
        
        ddiff_expr = f.datediff(f.col(self.date), f.col("prev_time")) ##difference in day
        w_o_expr = f.when(f.isnull(ddiff_expr), 0).otherwise(ddiff_expr)
        self.frame = self.frame.withColumn("diff_in_day", w_o_expr)
        
        magic_percentile = f.expr('percentile_approx(diff_in_day, 0.5)').alias('med_in_day')
        return self.frame.groupby([self.payer, self.recipient]).agg(magic_percentile)
    
    def __get_curr_stat(self):
        count_tr_expr = f.count(self.curr_sum).alias("freq")
        sum_tr_expr = f.round(f.sum(self.curr_sum),0).alias("summa")
        df_t = self.frame.groupby([self.payer, self.recipient]).agg(count_tr_expr, sum_tr_expr)
        
        ttl_sum_expr = f.sum("summa").alias("payer_total_sum")
        df_total_sum = df_t.groupby(self.payer).agg(ttl_sum_expr)
        
        df_t = df_t.join(df_total_sum, on=self.payer)
        pcnt_ttl_sum_expr = f.round((f.col("summa")/f.col("payer_total_sum"))*100, 2)
        return df_t.withColumn("%_of_total_sum", pcnt_ttl_sum_expr)
    
    def __preprocess(cls):
        cls.result = cls.__get_curr_stat().join(cls.__get_time_stat(), on=[cls.payer, cls.recipient])
        
        recipient_count_expr = f.count(cls.recipient).over(Window.partitionBy(cls.recipient))
        cls.result = cls.result.withColumn("r_count", recipient_count_expr)
        
        cls.result.cache()
        
class Painter:
    
    def __init__(self, init_graph):
        self.df = init_graph.result.toPandas()
        self.payer = init_graph.payer
        self.recipient = init_graph.recipient
        self.df_for_paint = None
        
    @staticmethod
    def get_values_for_graph(edgesess): 
        graph = {}
        for a, b in edgesess:
            if a not in graph:
                graph[a] = [b]
            else:
                graph[a].append(b)
        for a, b in edgesess:
            if b not in graph:
                graph[b] = [a]
            else:
                graph[b].append(a)
        return graph
    
    def coherence(cls, vertex, visited, temp_visited, graph):
        if vertex not in visited:
            visited.append(vertex)
            temp_visited.append(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    cls.coherence(neighbor, visited, temp_visited, graph)
        return visited

    def get_groups(cls, p_r_array):
        neighborhoods = cls.get_values_for_graph(p_r_array)
        nodes = set(p_r_array[:,0].tolist() + p_r_array[:,1].tolist())

        fully_con = []
        visited = []
        for vertex in nodes:
            temp_visited = []
            nodes_temp = cls.coherence(vertex, visited, temp_visited, neighborhoods)
            fully_con.append(temp_visited)
        return [i for i in fully_con if len(i) > 0]
    
    def filtering_df(cls, sum_tresh=0, r_count_tresh=0, acc_name=None):
        cls.df_for_paint = cls.df
        if acc_name is not None:
            groups = cls.get_groups(cls.df_for_paint[[cls.payer, cls.recipient]].values)
            try:
                group_with_acc_name = [i for i in groups if acc_name in i][0]
            except IndexError:
                print(f"The name <{acc_name}> not found")
                
            cls.df_for_paint = cls.df_for_paint[cls.df_for_paint[cls.payer].isin(group_with_acc_name) &\
                                        cls.df_for_paint[cls.recipient].isin(group_with_acc_name)]
        cls.df_for_paint = cls.df_for_paint[(cls.df_for_paint["summa"] > sum_tresh) &\
                                   (cls.df_for_paint["r_count"] > r_count_tresh)]

    def draw(self, filename='test', engine="sfdp", attributes={"repulsiveforce":"0.9", 
                                                               "K":"3",
                                                               "smoothing":"spring",
                                                               "levels":"2"}):
        
        if self.df_for_paint is None:
            self.df_for_paint = self.df
        edges = self.df_for_paint[["Payer", "Recipient"]].values
        counts = self.df_for_paint["freq"].values
        sums = self.df_for_paint["summa"].values.astype(int)
        percent = self.df_for_paint["%_of_total_sum"].values
        med_in_day = self.df_for_paint["med_in_day"].values
        
        graph = Digraph('G', 
                        filename=filename, 
                        engine=engine, 
                        graph_attr=attributes)
#         graph.attr(rank="same")

        for i in range(len(edges)):
            tr = edges[i]
            s = sums[i]
            p = percent[i]
            c = counts[i]
            d = med_in_day[i]
            graph.edge(f"{tr[0]}", f"{tr[1]}", label=f"f-{c} d-{d} s-{s} p-{p}")

        graph.view()