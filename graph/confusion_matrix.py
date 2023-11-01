import matplotlib.pyplot as plt

guess = ["Accident", "No Accident"]
fact = ["Accident", "No Accident"]
classes = list(set(fact))
classes.sort(reverse=True)
r1 = [[482, 18], [26, 474]]

plt.figure(figsize=(7, 5))  # 设置plt窗口的大小
confusion = r1
print("confusion", confusion)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
indices2 = range(3)
plt.xticks(indices, classes, fontsize=20)
plt.yticks([0.00, 1.00], classes, fontsize=20)
plt.ylim(1.5, -0.5)  # 设置y的纵坐标的上下限

plt.title("Confusion Matrix", fontdict={'weight': 'normal', 'size': 20})
# 设置color bar的标签大小
cb = plt.colorbar()
cb.ax.tick_params(labelsize=20)
plt.xlabel('Predict Label', fontsize=20)
plt.ylabel('True Label', fontsize=20)

print("len(confusion)", len(confusion))
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):

        if confusion[first_index][second_index] > 200:
            color = "w"
        else:
            color = "black"
        plt.text(first_index, second_index, confusion[first_index][second_index], fontsize=18, color=color,
                 verticalalignment='center', horizontalalignment='center', )
plt.tight_layout()
plt.savefig('./graduate/confusion_matrix.jpg')
plt.show()