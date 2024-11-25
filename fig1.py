# 可视化FC层，观察每个类别对应权重夹角
import matplotlib.pyplot as plt
import semilearn.nets as nets


num_classes = 10
pretrained = True
pretrained_path = 'saved_models/classic_cv_imb/fixmatch_disa_cifar10_lb1500_100_ulb3000_100_0/model_best.pth'
net_name = 'wrn_28_2'

builder = getattr(nets, net_name)
model = builder(num_classes=num_classes, pretrained=pretrained, pretrained_path=pretrained_path)

last_fc_weights = model.classifier.weight.data
weights = last_fc_weights.detach().numpy()  # 将PyTorch张量转换为numpy数组

print("Weights shape:", weights.shape)  # 输出weights的尺寸
print(weights)

# plt.imshow(weights, cmap='coolwarm')
# plt.colorbar()
# plt.title("Last Fully - Connected Layer Weights")
# plt.xlabel("Input Neurons")
# plt.ylabel("Output Neurons")
# plt.savefig('fig1.png')