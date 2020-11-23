import task3
import task5

def print_and_get_feedback(results):
    print("The results from Multi-dimensional nearest neighbor search task are:")
    i = 1
    for result in results:
        print(i, ". ", result)
        i = i + 1
    print("Please provide feedback for each result")
    feedback_list = list()
    i = 1
    for result in results:
        print(i, ". ", result)
        t = (result, int(input("Enter 1 for Relevant and 0 for Irrelevant: ")))
        feedback_list.append(t)
        i = i + 1
    return feedback_list


def main():
    while True:
        print("Task 3 Multi-dimensional index structures and nearest neighbor search task")
        results,t = task3.main()
        feedback = print_and_get_feedback(results)
        while True:
            task = int(input("Enter 4 for Probabilistic relevance feedback\nEnter 5 for Classifier-based relevance "
                             "feedback\nEnter 0 to exit"))
            if task == 4:
                print("Task 4 Probabilistic relevance feedback")
            if task == 5:
                task5.main(feedback,t)
                print("Task 5 Classifier-based relevance feedback")
            if task == 0:
                break
        v = int(input("Enter 1 to rewrite the layers and hashes\nEnter 0 to exit"))
        if v == 0:
            break


if __name__ == '__main__':
    main()