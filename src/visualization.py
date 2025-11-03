import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def plot_loss(losses, out_path):
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.figure()
	plt.plot(losses, label='train_loss')
	plt.xlabel('step')
	plt.ylabel('loss')
	plt.legend()
	plt.savefig(out_path)
	plt.close()

