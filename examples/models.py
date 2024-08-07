import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleDetector(nn.Module):
    def __init__(self):
        super(CircleDetector, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(in_features=64*16*16, out_features=128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=3)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        c_x, c_y, r = torch.tanh(x[:, 0]), torch.tanh(x[:, 1]), torch.sigmoid(x[:, 2])
        return c_x, c_y, r
class RectangleDetector(torch.nn.Module):
    def __init__(self):
        super(RectangleDetector, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(in_features=64*16*16, out_features=128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        t_x, t_y,b_x, b_y = torch.tanh(x[:, 0]), torch.tanh(x[:, 1]), torch.tanh(x[:, 2]), torch.tanh(x[:, 3])
        return t_x, t_y, b_x, b_y

class Inside(nn.Module):
    def __init__(self):
        super(Inside, self).__init__()

    def forward(self, c1, c2, r, xbl, ybl, xtr, ytr):
        smooth_lt_xbl = torch.sigmoid(10 * (c1 - (xbl + r)))
        smooth_gt_xtr = torch.sigmoid(10 * ((xtr - r) - c1))
        smooth_lt_ybl = torch.sigmoid(10 * (c2 - (ybl + r)))
        smooth_gt_ytr = torch.sigmoid(10 * ((ytr - r) - c2))
        return smooth_lt_xbl * smooth_gt_xtr * smooth_lt_ybl * smooth_gt_ytr

    def __call__(self, *args):
        return self.forward(*args)

class Outside(nn.Module):
    def __init__(self):
        super(Outside, self).__init__()

    def forward(self, c1, c2, r, xbl, ybl, xtr, ytr):
        smooth_gt_xbl = torch.sigmoid(10 * ((xbl - r) - c1))
        smooth_lt_xtr = torch.sigmoid(10 * (c1 - (xtr + r)))
        smooth_gt_ybl = torch.sigmoid(10 * ((ybl - r) - c2))
        smooth_lt_ytr = torch.sigmoid(10 * (c2 - (ytr + r)))
        return smooth_gt_xbl + smooth_lt_xtr + smooth_gt_ybl + smooth_lt_ytr - \
               smooth_gt_xbl * smooth_lt_xtr * smooth_gt_ybl * smooth_lt_ytr

    def __call__(self, *args):
        return self.forward(*args)

class Intersect(nn.Module):
    def __init__(self):
        super(Intersect, self).__init__()

    def forward(self, c1, c2, r, xbl, ybl, xtr, ytr):
        smooth_lt_xbl = torch.sigmoid(10 * (c1 + r - xbl))
        smooth_gt_xtr = torch.sigmoid(10 * (xtr - (c1 - r)))
        smooth_lt_ybl = torch.sigmoid(10 * (c2 + r - ybl))
        smooth_gt_ytr = torch.sigmoid(10 * (ytr - (c2 - r)))
        return smooth_lt_xbl * smooth_gt_xtr * smooth_lt_ybl * smooth_gt_ytr

    def __call__(self, *args):
        return self.forward(*args)

def compute_accuracy(circle_model, rect_model, inside_model, outside_model, intersect_model, test_dataloader, device):
    # Set models to evaluation mode
    circle_model.eval()
    rect_model.eval()
    inside_model.eval()
    outside_model.eval()
    intersect_model.eval()

    correct = 0
    total = 0

    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]

    with torch.no_grad():
        for batch_idx, (img_batch, label_batch) in enumerate(test_dataloader):
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            # Predict the circle parameters
            c_x, c_y, r = circle_model(img_batch)

            # Get the bounding box for the true circle
            t11, t12, b11, b12 = rect_model(img_batch)

            for i in range(len(img_batch)):
                # Extract the true one-hot encoded label
                true_class = int(label_batch[i].item())

                # Compute probabilities for being inside, outside, and intersecting
                inside_prob = inside_model(c_x[i], c_y[i], r[i], t11[i], t12[i], b11[i], b12[i])
                outside_prob = outside_model(c_x[i], c_y[i], r[i], t11[i], t12[i], b11[i], b12[i])
                intersect_prob = intersect_model(c_x[i], c_y[i], r[i], t11[i], t12[i], b11[i], b12[i])

                # Stack probabilities to form a tensor
                class_probs = torch.tensor([inside_prob, intersect_prob, outside_prob], device=device)
                predicted_class = torch.argmax(class_probs).item()

                # Update overall accuracy
                if predicted_class == true_class:
                    correct += 1
                    class_correct[true_class] += 1
                class_total[true_class] += 1
                total += 1

    accuracy = correct / total
    class_accuracies = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(3)]

    # Debug: print class-wise accuracies
    for i, class_acc in enumerate(class_accuracies):
        print(f"Accuracy for class {i}: {class_acc * 100:.2f}%")
    print(f"Overall accuracy: {accuracy * 100:.2f}%")

    # Set models back to training mode
    circle_model.train()
    rect_model.train()
    inside_model.train()
    outside_model.train()
    intersect_model.train()

    return accuracy, class_accuracies