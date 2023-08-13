from torch.utils.tensorboard import SummaryWriter
import math
from tqdm import tqdm

# Create a SummaryWriter instance for TensorBoard
writer = SummaryWriter()

num_steps = 5000

for idx_step in tqdm(range(num_steps)):
    X, label = datasets.make_moons(n_samples = 512, noise = 0.05)
    X = torch.Tensor(X).to(device = device)

    z, logdet = realNVP.inverse(X)

    loss = torch.log(z.new_tensor([2*math.pi])) + torch.mean(torch.sum(0.5*z**2, -1) - logdet)
    
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()

    if (idx_step + 1) % 1000 == 0:
        print(f"idx_steps: {idx_step:}, loss: {loss.item():.5f}")

    # Save images to TensorBoard every epoch
    if (idx_step + 1) % 1 == 0:
        X, label = datasets.make_moons(n_samples = 1000, noise = 0.05)
        X = torch.Tensor(X).to(device = device)
        z, logdet_jacobian = realNVP.inverse(X)
        z = z.cpu().detach().numpy()

        X = X.cpu().detach().numpy()
        fig = plt.figure(2, figsize = (12.8, 4.8))
        fig.clf()
        plt.subplot(1,2,1)
        plt.plot(X[label==0,0], X[label==0,1], ".")
        plt.plot(X[label==1,0], X[label==1,1], ".")
        plt.title("X sampled from Moon dataset")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")

        plt.subplot(1,2,2)
        plt.plot(z[label==0,0], z[label==0,1], ".")
        plt.plot(z[label==1,0], z[label==1,1], ".")
        plt.title("Z transformed from X")
        plt.xlabel(r"$z_1$")
        plt.ylabel(r"$z_2$")
        plt.savefig("image.png")
        image = plt.imread("image.png")
        image = torch.from_numpy(image.transpose((2, 0, 1))) # HWC to CHW
        writer.add_image("Generated Image", image, idx_step)

    # Save model checkpoint every 50 epochs
    if (idx_step + 1) % 50 == 0:
        torch.save(realNVP.state_dict(), f"realNVP_checkpoint_step_{idx_step}.pth")

plt.show()
