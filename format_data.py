import os, sys, shutil

from regex import D

dataset_name = input('Enter dataset folder name: ')
os.makedirs(dataset_name, exist_ok = True)
old_name  = input('Enter original folder name: ')
#existing_folder_num = input('How many example folders existed in the dataset folder: ')
old_name_array = ""
new_name_array = ""

if (os.path.isfile(dataset_name+'/OriginalFileName.txt')):
    my_file = open(dataset_name+'/OriginalFileName.txt',"r")
    content = my_file.read()
    print(content)
    content_list = content.split("\n")
    my_file.close()
    old_name_array = content_list[1]
    new_name_array = content_list[3]
    print(new_name_array)
    existing_folder_num = int(str(new_name_array)[-1])
    print(existing_folder_num)
else:
    existing_folder_num = -1
#new_name = input('Enter new folder name: ')

#folder_num = 0
if not (os.path.isfile(dataset_name+'/led_position_list_cart.npy')):
    shutil.copy2('led_position_list_cart.npy',dataset_name+'/led_position_list_cart.npy')
if not (os.path.isfile(dataset_name+'/optimalExposureRound.txt')):
    shutil.copy2(old_name+'/optimalExposureRound.txt',dataset_name+'/optimalExposureRound.txt')

'''
if (os.path.isdir(old_name+'/Multiplex_Random')):
    shutil.copytree(old_name+'/Multiplex_Random', dataset_name+'/training/example_00000'+str(int(existing_folder_num)+2), copy_function = shutil.copy)
    new_name_array = new_name_array+" example_00000"+str(int(existing_folder_num)+2)
    old_name_array = old_name_array+" "+old_name+"/Multiplex"
    #shutil.copytree(old_name+'/Multiplex', dataset_name+'/training/example_00000'+str(int(existing_folder_num)+2), copy_function = shutil.copy)
'''

#os.rename(old_name, new_name)
#os.makedirs(new_name+'/training/')
#shutil.copytree(old_name, dataset_name+'/example_00000'+str(int(existing_folder_num)+1), copy_function = shutil.copy)
shutil.copytree(old_name, dataset_name+'/training/example_00000'+str(int(existing_folder_num)+1), copy_function = shutil.copy)
new_name_array = new_name_array+" example_00000"+str(int(existing_folder_num)+1)
old_name_array = old_name_array+" "+old_name

    
with open(dataset_name+'/OriginalFileName.txt','w') as f:
    f.write('Original: \n')
    f.write(old_name_array+"\n")
    f.write('New: \n')
    f.write(new_name_array+"\n")

