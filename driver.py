import sys
import os

import task0
import task1
import task2
import task3
import task4
# import task5
import task5
import task6
import task7
# import task8
# import task9

task_map = [None, task1.start_task1, task2.start_task2, task3.start_task3, task4.start_task4, task5.start_task5,
            task6.start_task6, task7.start_task7]

if __name__ == '__main__':
    metadata_file = os.path.join(os.getcwd(), 'metadata.pickle')
    simp_file = os.path.join(os.getcwd(), 'simp.pickle')
    if not (os.path.isfile(metadata_file) and os.path.isfile(simp_file)):
        task0.start_task0(metadata_file, simp_file)

    inpt = ''
    while True:
        inpt = input('Enter task number (1-7, q to quit): ')
        if inpt == 'q':
            break

        task = int(inpt)
        if 1 <= task <= 7:
            task_map[task]()




        # try:
        #     task_map[task]()
        # except:
        #     print('Invalid task selection, please select from 1-9')