#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import skimage
import sys
import numpy as np
import cv2
import time
import os
import math
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv
import colorsys
import hasel
from PIL import Image
import pyhdust.images as phim
from threading import Thread, Lock
import logging

#Global vars

mutex = Lock()
results = []

colors = [[252,216,118],[246,147,68],[255,217,188],[236,150,89],[255,255,199],[255,249,227],[205,201,190],[255,254,250],[255,255,191],[255,254,197],[255,249,150],[255,255,251]]
colors2 =[[253, 255, 247],[251, 255, 250],[251, 255, 252],[254, 254, 254],[255, 253, 254],[255, 253, 252],[255, 253, 250],[255, 254, 250],[255, 253, 250],[255,255,255],[252, 254, 253],[254, 254, 254],[252, 254, 253],[232,137,91],[254,254,252],[254,225,149],[253,255,206],[252,216,118],[246,147,68],[255,217,188],[236,150,89],[255,255,199],[255,249,227],[205,201,190],[255,254,250],[255,255,191],[255,254,197],[255,249,150],[255,255,251],[254,253,199]]

actual_path = os.getcwd()

def prova():
	os.chdir(actual_path)

	src = cv2.imread("test.jpg",0) # read input imag
	
	cnts, hiers = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
	hull = []
	for i in range(len(cnts)):
		
		hull.append(cv2.convexHull(cnts[i], False))
	hull = np.asarray(hull)
	
	drawing = np.zeros((src.shape[0], src.shape[1]), np.uint8)
	# draw cnts and hull points
	for i in range(len(cnts)):
	    color_contours = (255, 255, 255) # green - color for cnts
	    color = (255, 255, 255) # blue - color for convex hull
	    # draw ith contour
	    cv2.drawContours(drawing, cnts, i, color_contours, 10, 10, hiers)
	    # draw ith convex hull object
	    #cv2.drawContours(drawing, hull, i, color, 1, 8)

	M = cv2.moments(drawing)
 
	# calculate x,y coordinate of center
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	print cX,cY
	#area = cv2.contourArea()
	#print area

	
	

	cv2.imshow('img',drawing)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
def makeVideoFromFrames():
	'''
	# per convertir de h264 a .mp4 : -> ffmpeg -framerate 24 -i bo4.h264 -c copy bo4.mp4
	# per grabar el video : -> aspivid -o bo4.h264 -t 30000
	'''
	os.system("ffmpeg -f image2 -r 10/1 -i ./frames/retall-bo-4-frame-hsv%01d.jpg -vcodec mpeg4 -y ./frames/result-hsv.mp4")
	return 0
def getFramesFromVid():
	os.chdir(actual_path)
	os.chdir('test')
	os.chdir('hsl')
	os.chdir('test4')
	
	print 'Iam: ',os.getcwd()
	
	
	

	# s'hauria d'executar la comanda ADD per passar els arrays vh264 (el video) a .mp4
	# os.execute()
	files = os.listdir(os.getcwd())
	print files
	print '\n'
	
	for file in files:
		namefile, extension = os.path.splitext(file)
		if (extension == '.mp4' or extension == '.MP4') and namefile == 'test':
			
			print 'filename: ',namefile
			# per grabar els frames del video vhs264 a fotos:
			count = 0
			vidcap = cv2.VideoCapture(file)
			
			
			success,image = vidcap.read()

			
			#hsv_img = rgb2hsv(image)
			#hsv_img = hsv_img[:,:,:] * 255
			# hue, saturation, lightness or luminance (color/to, saturació, brillantor)
			hsl_img = hasel.rgb2hsl(image)
			hsl_img = hsl_img[:,:,:] * 255
			
			while success:
				#hsv_img = rgb2hsv(image)
				#hsv_img = hsv_img[:,:,:]  * 255
				# hue, saturation, lightness or luminance (color/to, saturació, brillantor)
				hsl_img = hasel.rgb2hsl(image)
				hsl_img = hsl_img[:,:,:] * 255
				# cmyk
				#cmyk_img = phim.rgb2cmyk(np.asarray(img))
				# ycbcr
				#ycbcr_img = rgb2ycbcr(img)
				
				cv2.imwrite("%d00HSL.jpg" % count,hsl_img[:,:,:])
				
				cv2.imwrite("%d00HSL2.jpg" % count,hsl_img[:,:,2])
				

				#cv2.imwrite("%d-rgb.jpg" % count,image[:,:,:])

				#cv2.imwrite("hsllll1-%d.jpg" % count,hsl_img[:,:,1])
				#cv2.imwrite("hsllll0-%d.jpg" % count,hsl_img[:,:,0])

				#cv2.imwrite("estatic-frame-hsl%d.jpg" % count, hsl_img)     # save frame as JPEG file
				#cv2.imwrite("estatic-frame-hsv%d.jpg" % count, hsv_img)     # save frame as JPEG file      

				
				success,image = vidcap.read()
			
				count +=  1
			
				
			print 'Count: ',count	
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)
def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    print r,g,b
    mx = max(r.all(), g.all(), b.all())
    mn = min(r.all(), g.all(), b.all())
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

# Setup for testing algorithm

actual_path = os.getcwd()
os.chdir('test')
os.chdir('hsl')
os.chdir('test1')
os.chdir('hsl')

files = sorted(os.listdir(os.getcwd()))
files = [str(os.getcwd())+'/'+str(f) for f in files ]
#files = files[:5]

os.chdir(actual_path)
os.chdir('test')
os.chdir('hsl')
os.chdir('test1')
os.chdir('rgb')
files_rgb = sorted(os.listdir(os.getcwd()))
files_rgb = [str(os.getcwd())+'/'+str(f) for f in files_rgb ]
#files_rgb = files_rgb[:5]

os.chdir(actual_path)
os.chdir('test')
os.chdir('hsl')
os.chdir('test1')
os.chdir('hsl')


def RGB2HSL_Zones(recognizer_size_x=0,recognizer_size_y=0,white_tol=0.05,centroid_tol = 0.1, color_tol = 0.2):

	R = G = B = 255 
	
	size_x = 0
	size_y = 0

	segment = None
	
	
	frame_index = 0

	ini_x = 9999999
	ini_y = 9999999
	
	fi_x = 0
	fi_y = 0
	
	it = 0
	start = time.time()
	cX = 0
	cY = 0
	#imo = cv2.imread(files_rgb[0],1)
	#cv2.imshow('rgb original: ',imo)
	for file_hsl in files:
		namefile,extension = os.path.splitext(file_hsl)
		if extension == '.jpg':
			
			print 'RGB2HSL_Zones ',file_hsl,'...'
	
			photo = cv2.imread(file_hsl,0)
			#cv2.imshow('img original',photo)
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()
			
			max_width = photo.shape[1]
			max_height = photo.shape[0]
			
			max_num_colors = 0
			num_colors = 0

			start = time.time()
			last_is_white = False

			num_items_position = 0
			whites_pos_list = []
			
			
			for x in range(int(max_height/recognizer_size_x)+1):
				for y in range(int(max_width/recognizer_size_y)+1):
					try:
						
						if photo[ x * recognizer_size_x : (x * recognizer_size_x) + recognizer_size_x, 
							y * recognizer_size_y : (y * recognizer_size_y) + recognizer_size_y].shape != (max_height,max_width):
							
							segment = np.zeros(photo[ x * recognizer_size_x : (x * recognizer_size_x) + recognizer_size_x, y * recognizer_size_y : (y * recognizer_size_y) + recognizer_size_y].shape,dtype=np.uint8)
							
						else:
							segment = np.zeros((recognizer_size_x,recognizer_size_y),dtype=np.uint8)
						

						segment[:,:] = photo[ x * recognizer_size_x : (x * recognizer_size_x) + recognizer_size_x, 
							y * recognizer_size_y : (y * recognizer_size_y) + recognizer_size_y]

						size_x = segment.shape[0]
						size_y = segment.shape[1]

					except ValueError as e:
						pass
						
					nums = None
					num_counter = None
					freq_dic = None


					nums, num_counter = np.unique(segment[:,:], return_counts=True)
					freq_dic = dict(zip(nums,num_counter))

					try:
						if freq_dic[255]:
							#print 'Aquest bloc te blanc: '
							
							if freq_dic[255] >= ((size_y*size_x) * white_tol):
								#print 'Aquest bloc te blancs: ',freq_dic[255],' dun maxim de: ',(size_y*size_x) * 0.01

								num_colors = freq_dic[255]
								#print ' -> El bloc te mes de 0.1'
								if last_is_white:
									#print 'El bloc anterior a lactual es blanc'
									fi_x = (x*size_x) + size_x
									fi_y = (y*size_y) + size_y
								else:
									#print 'El bloc anterior a lactual NO es blanc'
									if ini_x > (x*size_x):
										ini_x = x*recognizer_size_x
										#print 'ini_x: ',ini_x
									
									if ini_y > (y*size_y):
										ini_y = y*recognizer_size_y
										#print 'ini_y: ',ini_y

									whites_pos_list.append([ini_x,ini_y])

									fi_x = (x*recognizer_size_x) + size_x
									fi_y = (y*recognizer_size_y) + size_y
								
								possiblyFire = True	
								last_is_white = True
							else:
								pass
								#print ' -> El bloc no te mes de 0.1'
							
					except KeyError:
						#print 'Bloc negre'

						if last_is_white:
							#print 'El bloc anterior a lactual es blanc, FINAL: ',ini_x,' - ',fi_x,'y_ini: ',ini_y, ' - ',fi_y
						
							whites_pos_list[-1].append(fi_x)
							whites_pos_list[-1].append(fi_y)
							whites_pos_list[-1].append(num_colors)

							num_colors = 0
							num_items_position += 1
							
							#print 'Whites pos list: ',whites_pos_list
							ini_x = 999999
							ini_y = 999999
							fi_y = 0
							fi_x = 0

							last_is_white = False
						else:
							
							ini_x = 999999
							ini_y = 999999
							fi_y = 0
							fi_x = 0
							#print 'El bloc anterior a lactual NO es blank'
					
			
			print 'Hiha ',num_items_position,' zones blanques'
			#print 'Whites pos: ',whites_pos_list
			#threads = [None] * num_items_position
			
			threads = []
			res = [None] * num_items_position
			
			#whites_pos_list = getBetterZones(whites_pos_list)
			num_new_zones = len(whites_pos_list)
			#print whites_pos_list

			#num_items_position = num_new_zones
			for zone in range(num_items_position):
				
				#print 'Analitzan la ',i, 'zona blanca'

				ini_x = whites_pos_list[zone][0]
				ini_y = whites_pos_list[zone][1]
				fi_x = whites_pos_list[zone][2]
				fi_y = whites_pos_list[zone][3]
				num_colors = whites_pos_list[zone][4]

				segment = photo[ini_x:fi_x,ini_y:fi_y]
								
				M = cv2.moments(segment)
			 
				# calculate x,y coordinate of center
				try:

					cY = int(M["m10"] / M["m00"])
					cX = int(M["m01"] / M["m00"])
					#print cY,cX
				except ZeroDivisionError:
					pass

				
				# Retall de blocs amb bastant negre

				new_segment = None
				in_y = 0
				in_x = 0
				f_y = 0
				f_x = 0

				if cY > ((recognizer_size_y / 2) + (recognizer_size_y*0.20)):
					in_y = ini_y+int(recognizer_size_y*0.25)
				else:
					in_y = ini_y

				if cY < ((recognizer_size_y / 2) + (recognizer_size_y*0.20)):
					f_y = fi_y-int(recognizer_size_y*0.25)
				else:
					f_y = fi_y

				if cX < ((recognizer_size_x / 2) + (recognizer_size_x*0.20)):
					f_x = fi_x-int(recognizer_size_y*0.25)
				else:
					f_x = fi_x

				if cX > ((recognizer_size_y / 2) + (recognizer_size_y*0.20)):
					in_x = ini_x + int(recognizer_size_y*0.25)
				else:
					in_x = ini_x

				

				# coordenades noves retallant negre

				ini_x = in_x
				ini_y = in_y
				fi_x = f_x
				fi_y = f_y
				new_segment = photo[in_x:f_x,in_y:f_y]

				
				
				# visualitzacio del centroide
				#segment[cX:cX+5,cY:cY+5] = 128
				#cv2.imwrite('Motion_ini_segment_'+str(zone)+'.jpg',segment)

				
				#print segment.shape
				cv2.imshow('Zone: '+str(zone),new_segment)
				#imm = cv2.imread(files_rgb[1],1)
				#cv2.imshow('Zone: '+str(zone),imm[ini_x:fi_x,ini_y:fi_y,:])
				

				
				if cX == 0 or cY == 0:
					print 'no zona'
				else:
					th = Thread(target=isFire, args = (ini_x,ini_y,fi_x,fi_y,files,recognizer_size_x,recognizer_size_y,white_tol,centroid_tol,color_tol,num_colors,zone,cX,cY))
					threads.append(th)
					th.start()
					
			for th in threads:
				th.join()
											
			it+=1
			break
		
	positives =0
	negatives=0
	end = time.time()
	print 'Time elapsed: ',end-start,' seconds.'

	print results
	cv2.waitKey(0)

	cv2.destroyAllWindows()

	
	'''for i in results:
		if i[1] == True:
			positives += 1
		elif i[1] == False:
			negatives += 1
		else:
			pass
	try:
		if positives == 0:
			print 'No hi ha foc: positives(',positives,') | negatives(',negatives,')'
		elif positives != 0 and negatives != 0:
			if positives / negatives > 1:
				print 'Hi ha foc: positives(',positives,') | negatives(',negatives,')'
			else:
				print 'No hi ha foc, pero potser error: positives(',positives,') | negatives(',negatives,')'
		elif negatives == 0:
			print 'Hi ha foc: positives(',positives,') | negatives(',negatives,')'

	except ZeroDivisionError:
		print 'Error ZeroDivision -> Hi ha foc: positives(',positives,') | negatives(',negatives,')'
		'''
def warmColorDetection(ini_x,ini_y,fi_x,fi_y,size_x,size_y,q,previous_num_whites,zone,it):
	'''
	os.chdir(actual_path)
	os.chdir('test')
	os.chdir('hsl')
	os.chdir('test_positiu4')
	os.chdir('rgb')'''
	
	
	positives = 0
	negatives = 0
 	local_results = []
	max_num_colors = (fi_x-ini_x)*(fi_y-ini_y)

	
	im = cv2.imread(files_rgb[it],1)
	segment = im[ini_x:fi_x, ini_y:fi_y,:] 
	#cv2.imwrite('Color_it'+str(it)+'_segment_'+str(zone)+'.jpg',segment)

	
	freq = None

	#cv2.imshow('zone: '+str(zone),segment)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	 
	p = 0
	n = 0
	for x in range(fi_x-ini_x):
		for y in range(fi_y-ini_y):
			R = segment[x,y,0]
			G = segment[x,y,1]
			B = segment[x,y,2]
			bgr = [B,G,R]
			#print bgr
			if bgr in colors2:
				p+=1
			else:
				n+=1
	#print 'Apareixen ',c,'/',max_num_colors,' colors calids.'
	#print 'Zona ',zone,':\nPositius: ',p,'/',max_num_colors,'\nNegatives: ',n,'/',max_num_colors,'\n'
	
	if p >= max_num_colors * q:
		
		return True
		
	else:
		
		return False		
def isFire(ini_x,ini_y,fi_x,fi_y,files,size_x,size_y,white_tol,centroid_tol, color_tol,previous_num_whites,zone,cxx,cyy):

	
	#print 'ini x: (',ini_x,') ini y: (',ini_y,') | fi x: (',fi_x,') fi y: (',fi_y,') ' 
	#print 'is Fire filename: ',img
	
	
	positives = 0
	positives_warm = 0
	negatives_warm = 0
	negatives = 0
 	frame = 0
 	it = 1
 	warm_results = []
	for img in files[1:]:
		namefile,extension = os.path.splitext(img)
		if extension == '.jpg':
			im = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
			max_num_colors = fi_y*fi_x
			num = 0
			R = 255
			G = 255
			B = 255
			segment = im[ini_x:fi_x, ini_y:fi_y]
			M = cv2.moments(segment)
			cY=0
			cX=0
		 	try:
				cY = int(M["m10"] / M["m00"])
				cX = int(M["m01"] / M["m00"])

				if (abs(cX-cxx) > int(fi_x-ini_x) * centroid_tol) or (abs(cY-cyy) > int(fi_y-ini_y) * centroid_tol)  :
					positives += 1
				else:
					negatives += 1
				
			except ZeroDivisionError:
				pass
			
			warm = warmColorDetection(ini_x,ini_y,fi_x,fi_y,size_x,size_y,color_tol,previous_num_whites,zone,it)
			
			if warm:
				positives_warm+=1
			else:
				negatives_warm+=1
		else:
			pass
		
		frame+=1
		it+=1
		
	try:
		mutex.acquire()
		results.append(('Zone '+str(zone),[positives,negatives],[positives_warm,negatives_warm]))
		mutex.release()

	except IndexError:
		# l'ultima foto no te on mirar, per tant error d'acces
		pass


def getBetterZones(zones_list):

	better_zones = [[] for i in range(len(zones_list))]
	better_zones[0] = zones_list[0]
	
	prev_x = zones_list[0][0]
	prev_y = zones_list[0][1]
	
	zones_diferent = 0


	for i,zone in enumerate(zones_list[1:]):
		if zone[1] == prev_y:
			#print 'La zona: ',zone
			better_zones[zones_diferent].append(zone)
			prev_y = zone[1]
		else:
			#print 'Zona diferent: ',zone
			zones_diferent+=1
			better_zones[zones_diferent].append(zone)
			prev_y = zone[1]
	
	zones_diferent += 1
	tmp = 0
	noves_zones = []
	
	for zones in better_zones[:zones_diferent]:
		if type(zones[0]) is list:
			start_zone_x = 0
			start_zone_y = 0
			end_zone_x = 0
			end_zone_y = 0

			noves_zones.append([zones[0][0],zones[0][1],zones[len(zones)-1][2],zones[len(zones)-1][3],0])
			tmp+=1
			
		else:
			noves_zones.append(zones)
			tmp+=1

	return noves_zones






	

if __name__ == '__main__':
	#makeVideoFromFrames()
	#getFramesFromVid()
	#RGB2HSL_Zones(int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]))
	#RGB2HSL_Zones(50,50,0.002,0.10,0.025)
	#RGB2HSL_Zones(25,25,0.0010,0.1,0.025)
	RGB2HSL_Zones(10,10,0.0010,0.1,0.025)
	#RGB2HSL_Zones(5,5,0.0010,0.1,0.025)








