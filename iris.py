import pandas

iris_df = pandas.read_csv("Iris.csv")

from sklearn.tree import DecisionTreeClassifier
from sklearn2pmml.pipeline import PMMLPipeline

iris_pipeline = PMMLPipeline([
    ("classifier", DecisionTreeClassifier())
])
iris_pipeline.fit(iris_df[iris_df.columns.difference(["Species"])], iris_df["Species"])

from sklearn2pmml import sklearn2pmml

sklearn2pmml(iris_pipeline, "DecisionTreeIris.pmml", with_repr=True)
