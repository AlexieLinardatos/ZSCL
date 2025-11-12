import matplotlib.pyplot as plt
import csv
import os
from matplotlib.ticker import FormatStrFormatter



def plot_metrics(path):
    x_label = []
    iterations = []
    accuracies_top1 = []
    accuracies_top5 = []
    

    for csv_file in os.listdir(path):
        if csv_file.endswith(".csv") and csv_file.find("metrics")!=-1:
            x_label.append(csv_file.split(".csv")[0].split("_")[-1])

            with open(path +"/"+ csv_file, newline="") as f:
                reader = list(csv.DictReader(f))

                iterations.append([int(row["iteration"]) for row in reader])
                accuracies_top1.append([float(row["top1"]) for row in reader])
                accuracies_top5.append([float(row["top5"]) for row in reader])
                # ZSCL_losses = [row["ZSCL"] for row in reader]
    
    # print(x_label)
    # print(iterations)
    # print(accuracies_top1)
    # print(accuracies_top5)
    # print(ZSCL_losses)

    index = 0
    for dataset in x_label:
        plt.plot(iterations[index], accuracies_top1[index], label=dataset)
        
        index+=1

    # plt.ylim(0,100)
    plt.title("accuracy over iterations\n \
            Trained on: DTD",
            fontsize=12)
    plt.legend()
    plt.ylabel("accuracy(%)")
    plt.xlabel("iterations")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    plt.savefig(path +"/"+ "output.png")

    # plt.plot(iterations, )

def plot_all(path):
    versions = []
    datasets = []
    accuracies_top1 = []
    accuracies_top5 = []

    index = 1
    while True:
        folder = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

        if folder:
            folder = folder[0]
            print(f"folder:{folder}")
        
        for csv_file in os.listdir(path):
            if csv_file.endswith(".csv") and csv_file.find("results")!=-1:
                print(csv_file)
                versions.append(path.split("/")[-1])

                with open(path +"/"+ csv_file, newline="") as f:
                    reader = list(csv.DictReader(f))

                    datasets = [(row["dataset"]) for row in reader]
                    accuracies_top1.append([float(row["top1"]) for row in reader])
                    accuracies_top5.append([float(row["top5"]) for row in reader])
        index+=1
        if not folder:
            break
        path = os.path.join(path,folder)
    
    versions[0] = "DTD"
    print(datasets)
    print(accuracies_top1)    
    print(versions)

    index = 0
    for dataset in datasets:
        plt.plot(versions, [accuracies_top1[i][index] for i in range(len(versions))], label=dataset)
        
        index+=1

    # plt.ylim(0,100)
    plt.title("accuracy over iterations\n \
            Trained on: DTD",
            fontsize=12)
    plt.legend()
    plt.ylabel("accuracy(%)")
    plt.xlabel("versions")
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    plt.savefig(path +"/"+ "output_all.png")

if __name__ == "__main__":
    plot_all("./ckpt/DTD/")