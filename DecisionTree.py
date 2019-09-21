from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

# Load Iris Dataset
iris = load_iris()

# Assign a DecisionTreeClassifier to clf (the criterion is entropy)
clf = tree.DecisionTreeClassifier(criterion="entropy")

# Plot the Decision Tree with the iris Dataset
tree.plot_tree(clf.fit(iris.data, iris.target))

# Predict the Class of the following Dataset, based on the created DecisionTree
result = clf.predict([[1.0, 0.2, 0.7, 0.4]])
print(result)

# Create PDF from Decision Tree and name it "DTree"
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("DTree")