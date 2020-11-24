import task1
import task2_knn
import task2_ppr
import task3
import task6


def read_directory():
    with open("directory.txt", 'r') as f:
        param = f.read()
        f.close()
    return param


def store_directory(directory):
    with open("directory.txt", 'w') as f:
        f.write(directory)
        f.close()


def main():
    while True:
        file = input("Input the directory with gestures")
        store_directory(file)
        print("########## Phase 3 ##########")
        print("Task 1: Personalized Pagerank")
        print("Task 2a: KNN classification")
        print("Task 2b: PPR classification")
        print("Task 3: Find similar gestures using LSH")
        print("Task 6: Relevance Feedback Interface")
        task = input("What Task do you want to perform: (enter 0 to exit)\n")
        # folder_directory = input("Input Directory path of the gesture folders:\n")
        if task == '1':
            print("########## Task 1 ##########")
            task1.main()
            print("########## Task 1 Completed ##########\n")
        elif task == '2a':
            print("########## Task 2 ##########")
            task2_knn.main()
            print("########## Task 2a Completed ##########\n")
        elif task == '2b':
            print("########## Task 2 ##########")
            task2_ppr.main()
            print("########## Task 2b Completed ##########\n")
        elif task == '3':
            print("########## Task 3 ##########")
            task3.main()
            print("########## Task 3 Completed ##########\n")
        elif task == '6':
            print("########## Task 6 ##########")
            task6.main()
            print("########## Task 6 Completed ##########\n")
        elif task == '0':
            exit()


if __name__ == '__main__':
    main()
