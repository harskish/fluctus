from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, abspath, join
from glob import glob
import sys
import csv

def listFiles(path):
    files = sorted(glob(join(path, '*.csv')))
    return files

# Format:
# {
#     benchmark1: {
#         scene1: {
#             'time': [],
#             'primary': [],
#             'extension': [],
#             'shadow': [],
#             'total': [],
#             'samples': [],
#         },
#         scene2: {...},
#         scene3: {...}
#     },
#     benchmark2: {
#         scene2: {...},
#         scene3: {...}
#     }
# }

data = {}
scene_names = set()

if __name__ == '__main__':
    files = listFiles(abspath('.'))
    if len(sys.argv) > 1:
        files = sys.argv[1:]

    for benchmark in files:
        with open(benchmark, 'r') as csvfile:
            print("Processing file: " + benchmark)
            data[benchmark] = {}
            plots = csv.reader(csvfile, delimiter=';')
            current_scene = ''
            for i, row in enumerate(plots):
                if i == 0:
                    header = row[1:] # indices offset by one
                    continue

                scene = row[0].replace('\\', '/').split('/')[-1]
                if current_scene != scene:
                    scene_names.add(scene)
                    data[benchmark][scene] = {}
                    for h in header:
                        data[benchmark][scene][h] = []
                    current_scene = scene
                for idx, value in enumerate(row[1:]):
                    data[benchmark][scene][header[idx]].append(float(value))

    wanted_datas = ['total'] #, 'primary', 'extension', 'shadow']

    for scene in scene_names:
        print("Creating plot for " + scene)
        fig = plt.figure()
        plt.title(scene)
        axes = fig.add_subplot(111)
        
        for benchmark_name in sorted(data.keys()):
            benchmark = data[benchmark_name]
            if scene not in benchmark:
                continue
            
            scene_data = benchmark[scene]
            for colname in wanted_datas:
                if colname not in scene_data:
                    continue
                axes.plot(scene_data['time'][1:], scene_data[colname][1:], label="{}: {}".format(benchmark_name, colname))

        plt.legend(loc='upper left');

    plt.show()