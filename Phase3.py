import task1


def main():
    while True:
        print("########## Phase 3 ##########")
        print("Task 1: Personalized Pagerank")
        task = input("What Task do you want to perform: (enter 0 to exit)\n")
        # folder_directory = input("Input Directory path of the gesture folders:\n")
        if task == '1':
            print("########## Task 1 ##########")
            task1.main()
            print("########## Task 1 Completed ##########\n")

        elif task == '0':
            exit()


if __name__ == '__main__':
    main()
