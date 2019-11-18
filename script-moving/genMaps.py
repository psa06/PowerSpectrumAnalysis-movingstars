import numpy as np
import glob, os
import shutil
import time

path = os.getcwd()
cmd = "sbatch job.s"

def deleteContent(open_file):
	open_file.seek(0)
	open_file.truncate()

maps_folder = path + "/movingstars_magmaps"
positions_folder = path + "/positions_extracted"

#create folder for the maps
if os.path.exists(maps_folder):
	print("folder already exists.")
else:
	os.mkdir(maps_folder)


#number of maps that are going to be outputted
N = len([name for name in os.listdir(positions_folder)])
print("number of maps: ", N)

positions = sorted(glob.glob(positions_folder+"/*.dat"), key=lambda f: int(filter(str.isdigit,f)))

for i,pos in enumerate(positions):
	#remove the already existing map	
	if os.path.exists(path+"/map.fits"):
		os.remove(path+"/map.fits")
	else:
		print(" ")

	#place the generated positions in lens_pos.dat after emptying it
	f_from = open(pos, 'r')	
	
	f_to = open(path+"/lens_pos.dat", 'w')	
	deleteContent(f_to)
	f_to.write(f_from.read())
	
	f_from.close()
	f_to.close()

	f_to = open(path+"/lens_pos.dat", 'r')
		
	#submit job using the new generated positions
	returned_value = os.system(cmd)
	time.sleep(30) #wait for 10 seconds	

	#Rename and store map in maps_folder
	new = "map%s.fits" %str(i)
	os.rename("map.fits", new)
	shutil.move(new , maps_folder+"/"+new)
	print(i)
















