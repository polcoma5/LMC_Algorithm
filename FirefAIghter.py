import skimage
import numpy as np
import cv2
import time
import os
import math
from skimage.color import rgb2hsv
from PIL import Image
from threading import Thread, Lock
import logging
from picamera import PiCamera
from picamera.array import PiRGBArray
import picamera.array
import RPi.GPIO as GPIO   
import BBTree as BBT 
import LineFollower as LF
import hasel
#import sensor_distancia as SD
#import Codigo_controlador_motor as MOT
#from sensor_distancia import *

colors2 =[[253, 255, 247],[251, 255, 250],[251, 255, 252],[254, 254, 254],[255, 253, 254],[255, 253, 252],[255, 253, 250],[255, 254, 250],[255, 253, 250],[255,255,255],[252, 254, 253],[254, 254, 254],[252, 254, 253],[232,137,91],[254,254,252],[254,225,149],[253,255,206],[252,216,118],[246,147,68],[255,217,188],[236,150,89],[255,255,199],[255,249,227],[205,201,190],[255,254,250],[255,255,191],[255,254,197],[255,249,150],[255,255,251],[254,253,199]]

class FirefAIghter():

	def __init__(self,widht=0,height=0):
		# Init motores
		self.in1 = 26
		self.in2 = 19
		self.in3 = 21
		self.in4 = 29
		self.ena = 12 
		self.enb = 13
		GPIO.setmode(GPIO.BCM)
		GPIO.setup(self.in1,GPIO.OUT)
		GPIO.setup(self.in2,GPIO.OUT)
		GPIO.setup(self.in3,GPIO.OUT)
		GPIO.setup(self.in4,GPIO.OUT)
		GPIO.setup(self.ena,GPIO.OUT)
		GPIO.setup(self.enb,GPIO.OUT)
		GPIO.output(self.in1,GPIO.LOW)
		GPIO.output(self.in2,GPIO.LOW)
		GPIO.output(self.in3,GPIO.LOW)
		GPIO.output(self.in4,GPIO.LOW)
		self.potmotl = GPIO.PWM(self.ena,1000)  
		self.potmotr = GPIO.PWM(self.enb,1000)
		self.potmotl.start(100)
		self.potmotr.start(100)
		self.potmotln = 90
		self.potmotrn = 90
		
		#Init camera
		self.picamera = PiCamera()
		self.picamera.resolution = (640,480)
		self.arbol = BBT.Stack()
		self.isFire = None # variable global que indica quan es troba foc
		self.results_color_detection = None
		self.results_motion_detection = None
		self.mutex = Lock()
		self.recognizer_size_x = 40
		self.recognizer_size_y = 80
		self.white_tolerance = 0.05
		self.color_tolerance = 0.25
		

		self._main()

	def _main(self):
		while 1:
			op = raw_input('Introduce la opcion que quieres ejecutrar:\n1.- Reconocimiento de zonas.\n2.- Ir a una zona.\n3.- MoverMotores\n')
			if op == '1':
				self._lineFollower(op='store')
			elif op == '2':
				self._lineFollower(op='goto',zone=0)
			elif op == '3':
				self._moveMotors()
			elif op =='4':
				self.identifyWhiteZones()

    	# escull una opcio:
		#	1.- (proces 1) reconeixer zones dela fabrica (linefollower y guardarCami) quan acaba la funcio 
		#		es torna al main, perque hi ha el while true.

		#	2.- (proces 2) anar a una zona concreta (op 1 s'ha dhaver executat abans) (linefollower) 
		#		
		#		m'entre s'executa la opcio 2:

		#			2.1- Linefollower per arribar a la zona i despres:

		#				A.- (thread 1, del proces 2) Detectar foc

		#					2.1.1- Situarse devant del foc, situar la maneguera i apagar.

		#				B.- (thread 2, del proces 2) Moviment per la zona

		#				### tant A com B s'executan a la vegada, A pot parar en qualsevolmoment 
						### al thread 1.2 per apagar foc ##
			
	#########################################################################################################################
	#########################################################################################################################
	# op: 'store' executa el linefollowe i va guardant a l'abre binari
	# op: 'goto' executa el linefollower i va cap a la zona.
	#########################################################################################################################
	def _lineFollower(self,op='',zone=None):

		# poner el codigo que tengais aqui

		# diria que ahora solo queda programar que el robot empiece a desplazarse hacia delante i vaya tomando imagenes 
		# para la ver la linia, a la vez que va girando los motores segun diga el follower.

		# linefollower retorna q hay que girar a la derecha 90, _moveMotors(programa para mover )

		self.picamera.resolution = (426,240)
		rawCapture = PiRGBArray(self.picamera)
		self.picamera.start_preview()
		frame_rate = 10
		
		prev = 0
		crearMapa = False
		CountIrLlama = 0
		irFuego = False
		aux = None

		
		if op == 'store':

			crearMapa = True #si hay que crear el mapa
			irFuego = False
		else:

			irFuego = True #si hay que ir a un sector
			sectorLlamas = zone #que sector se quema
			CountIrLlama = 1 #contador del camino al fuego

		# Inicializar arbol
		if crearMapa:
		    CountBif=0
		    CountSec=0
		    pilaBif = BBT.Stack()
		    Sectores = BBT.Stack()
		    root = BBT.Node(CountBif)
		    aux = BBT.Node(1)
		    aux = root
		
		for image in self.picamera.capture_continuous(rawCapture, format="bgr"):
		    frame = image.array
		    ## Capturar frame a frame
		    time_elapsed = time.time() - prev
		    #ret, frame = cap.read()
		    bif=0 #si bifurcacion 1
		    sec=0 #si sector 1
		    
		    if time_elapsed > 1./frame_rate:
		        
		        prev = time.time()
		    	
		        
		        image.truncate(0)
		        #crop_img = seguirLinea(io.imread("r.jpeg"))
		        crop_img,bif,sec,mappedCx = LF.seguirLinea(frame)

		        #######
		        #dist, distpx = SD.distance()
		        #######

		        #print ("Distancia: ",dist," Distnacia pixel: ",distpx)
                        #cv2.line(frame,(214,240),(214,int(distpx)),(255,0,255),1)
                        #cv2.imwrite("distancia.jpg", frame)	

                #######	        
		        """if dist <= 45:
                            #parar
		            self._moveMotors(0,0,0,0,0,0,0)
		            print ("obstaculo")
                            pass"""
                ########

		        #cv2.imwrite("fotico.jpg", crop_img)
		        #cv2.imwrite("fotico2.jpg", crop_img)
		        #fotico=cv2.imread("fotico.jpg")

		        #######
		        #cv2.imshow('frame',crop_img)
		        #cv2.waitKey(0) #to raro
		        #######

		        #time.sleep(3)
		        #cv2.destroyAllWindows()
		        
		        ### Ir a un sector
		        if irFuego:
		        	p=0
		        	if mappedCx < 0:
		        		p=self.potmotln*((100-abs(mappedCx))/100)
		        		self._moveMotors(1,0,0,1,90,self.potmotrn,1)
		        	elif mappedCx >= 0:
		        		p=self.potmotrn*((100-abs(mappedCx))/100)
		        		self._moveMotors(1,0,0,1,self.potmotln,90,1)
		        		#if bif==1: # BIFURACION
		        		#hacer que gire en la direccion que diga
		        		#cuando detecta distancia entre bif y robot de 32cm
		        		#avanzar unos 20-25cm y girar 90 grados si hay giro(decidir en que direccion iran las bifus(si izquierda o derecha))
		        		"""
		        		if Sectores.stack[sectorLlamas][CountIrLlama] == aux.left.data:
		        		p=0
		        		self._moveMotors(1,0,0,1,self.potmotl,self.potmotr,1)time.delay(1)
		        		self._moveMotors(0,0,0,1,p,self.potmotr,1)
		        		elif Sectores.stack[sectorLlamas][CountIrLlama] == aux.right.data:
		        		if mappedCx < 0:
		        		p=self.potmotl*((100-abs(mappedCx))/100)
		        		self._moveMotors(1,0,0,1,p,self.potmotr,1)
		        		elif mappedCx >= 0:
		        		p=self.potmotr*((100-abs(mappedCx))/100)self._moveMotors(1,0,0,1,self.potmotl,p,1)"""########
		        	if bif==1:
		        		p=0
		        		self._moveMotors(1,0,0,1,self.potmotln,self.potmotrn,1)
		        		time.delay(1)
		        		self._moveMotors(0,0,0,1,p,self.potmotrn,1)
		        		time.delay(1.5)
		        		self._moveMotors(0,0,0,0,0,0,1)
		        	elif sec==1:
		        		if Sectores.stack[sectorLlamas][CountIrLlama] == aux.left.data:
		        			hola=0 #cambiar modo a apagar fuegos y tal
		        		elif Sectores.stack[sectorLlamas][CountIrLlama] == aux.right.data:
		        			hola=0
		        		else:
							if mappedCx < 0:
								p=self.potmotl*((100-abs(mappedCx))/100)
								self._moveMotors(1,0,0,1,p,self.potmotr,1)
							elif mappedCx >= 0:
								p=self.potmotr*((100-abs(mappedCx))/100)
								self._moveMotors(1,0,0,1,self.potmotl,p,1)
		        ###
		        
		        ### Crear mapa
		        if crearMapa==1:
		            if bif==1:
		                CountBif = CountBif + 1
		                aux = aux.RecorrerMapa(CountBif)
		                pilaBif.pushh(CountBif)
		            if sec==1: # SECTOR
		                CountBif = CountBif + 1
		                aux.data = CountBif
		                pilaBif.pushh(CountBif)
		                auxSec = pilaBif.stack.copy()
		                Sectores.pushh(auxSec)
		                CountSec = CountSec + 1
		                aux = aux.padre
		                pilaBif.popp()
		            root.PrintTree()
		        ###
		        
		            ####

		            ####
		            ## Guardar frame en color o gris
		            #cv2.imwrite(time.strftime("%Y%m%d-%H%M%S"), frame)
		            #cv2.imwrite(time.strftime("%Y%m%d-%H%M%S"), gray)
		            ####

		            ####
		            ## Mostrar frame
		            #cv2.imshow('frame',gray)
		            #if cv2.waitKey(1) & 0xFF == ord('q'):
		            #	break
		            ####
		            

		#cap.release()
		cv2.destroyAllWindows()
	
	#########################################################################################################################
	#########################################################################################################################
	# self.motor_dret: 0
	# self.motor_esq: 1

	# Exemples:
	# programa 1: [0,55] motor dret (0) a (55) 
	# programa 2: [1,55] motor esq (1) a (55)		   
	# programa 3: [2,55,10] moure(1), els dos motors(2), dret a 55, esq a 10

	# potser tambe faltaria dirli el temps que volem que tingui els motors girant no?
	#########################################################################################################################
	def _moveMotors(self,in1,in2,in3,in4,ena,enb,t):
		print ('moviendo motores')
		self.potmotl.ChangeDutyCycle(ena)
		self.potmotr.ChangeDutyCycle(enb)
		GPIO.output(self.in1,in1)
		GPIO.output(self.in2,in2)
		GPIO.output(self.in3,in3)
		GPIO.output(self.in4,in4)
		#moverse por fps
		"""GPIO.output(self.in1,0)
		GPIO.output(self.in2,0)
		GPIO.output(self.in3,0)
		GPIO.output(self.in4,0)"""
		# program: ej: [2,55,10] moure els dos motors(2), dret a 55, esq a 10
		

	#########################################################################################################################
	#########################################################################################################################
	# obtenemos valor del sensor
	def _getProximity(self):
		pass     
		# return self.proximitat.getvalue() me lo invento
	
	#########################################################################################################################
	#########################################################################################################################
	# obtenemos valor del sensor
	def _getLuminosity(self):
		pass
		# return self.lluminositat.getvalue() me lo invento

	def identifyWhiteZones(self):
		self.picamera.resolution = (640,480)
		rawCapture = PiRGBArray(self.picamera)
		self.picamera.start_preview()
		self.picamera.framerate = 10
	
		# self move motors, detect limits, go in quadratic zone.
		
		for image in self.picamera.capture_continuous(rawCapture, format="bgr"):
		    
		    # Aquesta variable s'actualitza un cop sha detectat zones blanques i s'analitzen en el motion i color.
		    
		    if self.isFire:
		    	print 'Hay fuego'
		    	print self.results_color_detection
		    	print self.results_motion_detection
		    	break

		    image.truncate(0)
		    cv2.imwrite('tmp_id.jpg',image.array)

		    np_arr_rgb = cv2.imread('tmp_id.jpg',1)
		    photo = hasel.rgb2hsl(np_arr_rgb)
		    photo[:,:,:] = photo[:,:,:] * 255
		    photo[:,:,:] = np.uint8(photo[:,:,:])
		    black = photo[:,:,2]

		    #cv2.imshow('img',np.uint8(photo))
		    #cv2.waitKey(0)
		    #cv2.destroyAllWindows()
		  
		    R = G = B = 255 
		    size_x = 0
		    size_y = 0
		    segment = None
		    ini_x = 9999999
		    ini_y = 9999999
		    fi_x = 0
		    fi_y = 0
		    cX = 0
		    cY = 0                
		    max_width = photo.shape[1]
		    max_height = photo.shape[0]
		    max_num_colors = 0
		    num_colors = 0
		    start = time.time()
		    last_is_white = False
		    possiblyFire = False
		    num_white_zones = 0
		    whites_pos_list = []
		    
		    # Analyse frame by segments, getting only white zones 
		    for x in range(int(max_height/self.recognizer_size_x)+1):
		    	for y in range(int(max_width/self.recognizer_size_y)+1):
		    		try:
		    			if black[ x * self.recognizer_size_x : (x * self.recognizer_size_x) + self.recognizer_size_x, y * self.recognizer_size_y : (y * self.recognizer_size_y) + self.recognizer_size_y].shape != (max_height,max_width):
		    				segment = np.zeros((photo[ x * self.recognizer_size_x : (x * self.recognizer_size_x) + self.recognizer_size_x, y * self.recognizer_size_y : (y * self.recognizer_size_y) + self.recognizer_size_y].shape[0], photo[ x * self.recognizer_size_x : (x * self.recognizer_size_x) + self.recognizer_size_x, y * self.recognizer_size_y : (y * self.recognizer_size_y) + self.recognizer_size_y].shape[1]),dtype=np.uint8)	
		    			else:
		    				segment = np.zeros((self.recognizer_size_x,self.recognizer_size_y),dtype=np.uint8)

		    			segment[:,:] = black[ x * self.recognizer_size_x : (x * self.recognizer_size_x) + self.recognizer_size_x, y * self.recognizer_size_y : (y * self.recognizer_size_y) + self.recognizer_size_y]
		    			size_x = segment.shape[0]
		    			size_y = segment.shape[1]
		    			#cv2.imshow('img',np_arr_rgb[x * self.recognizer_size_x : (x * self.recognizer_size_x) + self.recognizer_size_x, y * self.recognizer_size_y : (y * self.recognizer_size_y) + self.recognizer_size_y,:])
		    			#cv2.waitKey(0)
		    			#cv2.destroyAllWindows()
		    			
		    		except ValueError as e:
		    			raise e

		    		nums = None
		    		num_counter = None
		    		freq_dic = None
		    		nums, num_counter = np.unique(segment[:,:], return_counts=True)
		    		freq_dic = dict(zip(nums,num_counter))

		    		try:
		    			if freq_dic[255]:
		    				if freq_dic[255] >= ((size_y*size_x) * self.white_tolerance):

		    					num_colors = freq_dic[255]
		    					possiblyFire = True
		    					if last_is_white:
		    						fi_x = (x*size_x) + size_x
		    						fi_y = (y*size_y) + size_y
		    					else:
		    						if ini_x > (x*size_x):
		    							ini_x = x*self.recognizer_size_x
		    						if ini_y > (y*size_y):
		    							ini_y = y*self.recognizer_size_y
		    						whites_pos_list.append([ini_x,ini_y])
		    						fi_x = (x*self.recognizer_size_x) + size_x
		    						fi_y = (y*self.recognizer_size_y) + size_y
		    						
		    					last_is_white = True
		    				else:
		    					pass            
		    		except KeyError:
		    			if last_is_white:
		    				whites_pos_list[-1].append(fi_x)
		    				whites_pos_list[-1].append(fi_y)
		    				whites_pos_list[-1].append(num_colors)
		    				num_colors = 0
		    				num_white_zones += 1
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
		    
		    if possiblyFire:
		    	threads = []
		    	res = [None] * num_white_zones
		    	self.results_motion_detection = [[] for i in range(num_white_zones)]
		    	self.results_color_detection = [[] for i in range(num_white_zones)]
		    	print 'White zones: ',num_white_zones
		    	
		    	for zone in range(num_white_zones):
		    		
		    		ini_x = whites_pos_list[zone][0]
		    		ini_y = whites_pos_list[zone][1]
		    		fi_x = whites_pos_list[zone][2]
		    		fi_y = whites_pos_list[zone][3]
		    		num_colors = whites_pos_list[zone][4]
		    		segment = black[ini_x:fi_x,ini_y:fi_y]

		    		#print '1.- Identify SHAPE centroide zona: ',zone,' shape: ',segment.shape
		    		M = cv2.moments(segment)
		    		
		    		try:
		    			cY = int(M["m10"] / M["m00"])
		    			cX = int(M["m01"] / M["m00"])
		    			
		    		except ZeroDivisionError:
		    			print 'ZeroDivisionError analitzant la zona ',zone

		    		
		    		# retall de segments bastant buits utilitzan les coordenades del centroide blanc
		    		new_segment = None
		    		if cY > ((self.recognizer_size_y / 2) + (self.recognizer_size_y*0.20)):
		    			ini_y += int(self.recognizer_size_y*0.25)
					if cY < ((self.recognizer_size_y / 2) + (self.recognizer_size_y*0.20)):
						fi_y -= int(self.recognizer_size_y*0.25)
					if cX < ((self.recognizer_size_x / 2) + (self.recognizer_size_x*0.20)):
						fi_x -= int(self.recognizer_size_y*0.25)
					if cX > ((self.recognizer_size_y / 2) + (self.recognizer_size_y*0.20)):
						ini_x += int(self.recognizer_size_y*0.25)
						
					segment = black[ini_x:fi_x,ini_y:fi_y]
		    		M = cv2.moments(segment)
		    		cv2.imshow('img',np_arr_rgb)
		    		cv2.waitKey(0)
		    		cv2.destroyAllWindows()
		    				    		
		    		try:
		    			cY = int(M["m10"] / M["m00"])
		    			cX = int(M["m01"] / M["m00"])
		    		except ZeroDivisionError:
		    			print 'Identify: ZeroDivisionError analitzant la zona: ',zone
		    		
		    		if cX == 0 or cY == 0:
		    			print 'El centroide surt 0,0 per tant no analitzem aquesta zona. Deu ser redundant'
		    		else:
		    			#th = Thread(target = self.foo, args =())
		    			self.run(ini_x,ini_y,fi_x,fi_y,num_colors,zone,cX,cY)
		    			#th = Thread(target = self.run, args =(ini_x,ini_y,fi_x,fi_y,num_colors,zone,cX,cY))
		    			#threads.append(th)
		    			#th.start()
		    			#th.join()
				#for th in threads:
				#	th.join()
			else:
				print 'En este frame no hay fuego'

	def run(self,ini_x,ini_y,fi_x,fi_y,previous_num_whites,zone,cxx,cyy):
		#self.picamera.resolution = (640,480)

		rawCapture = PiRGBArray(self.picamera)
		
		self.picamera.start_preview()
		
		#self.picamera.framerate = 3
		
		tmp_res = []

		iterations = 0
	
		for image in self.picamera.capture_continuous(rawCapture, format="bgr"):
			
			if iterations == 2:
				print 'End motion thread zone: ',zone
				break

			cv2.imwrite('tmp_isF.jpg',image.array)
			np_arr_rgb = cv2.imread('tmp_isF.jpg')
			photo = hasel.rgb2hsl(np_arr_rgb)
			photo[:,:,:] = photo[:,:,:] * 255
			photo[:,:,:] = np.uint8(photo[:,:,:])
			black = photo[:,:,2]
			
			positives = 0
			negatives = 0
			frame = 0
			warm_results = []
			image.truncate(0)
			max_num_colors = fi_y*fi_x
			num = 0
			cX = 0
			cY = 0
			segment = black[ini_x:fi_x, ini_y:fi_y]
			
			'''freq = None
			nums, num_counter = np.unique(segment[:,:], return_counts=True)
			freq = dict(zip(nums, num_counter))
			try:
				num = freq[255]
				# Provar aquesta comparacio amb un or (abs(previous_num_whites - freq[255])) < previous_num_whites*q
				if ((abs(previous_num_whites - num)) > previous_num_whites*self.white_tolerance):
					positives += 1
				else:
					negatives += 1
			except KeyError:
				negatives += 1'''

			M = cv2.moments(segment)
			
			try:
				cY = int(M["m10"] / M["m00"])
				cX = int(M["m01"] / M["m00"])
				#print 'Anteriors: ',cxx,cyy,' actuals: ',cX,cY
				#print 'Diferencia | x: ',abs(cX-cxx),' y: ',abs(cY-cyy)

				if (abs(cX-cxx) > int(fi_x-ini_x) * self.white_tolerance) :
					#print 'es mou bastan a les x'
					positives += 1
				else:
					negatives += 1
					#print 'No es mou gaire a les x'

				if (abs(cY-cyy) > int(fi_y-ini_y) * self.white_tolerance):
					#print 'es mou bastan a les y'
					positives += 1
				else:
					#print 'No es mou gaire a les y'
					negatives += 1
			
			except ZeroDivisionError:
				print 'isFire  -> ZeroDivisionError analitzant la zona: ',zone

			tmp_res.append(['Zona: '+str(zone),positives,negatives])

			self.results_motion_detection[zone].append(['Zona: '+str(zone),positives,negatives])
		
			w_result = self.warmColorDetection(np_arr_rgb,ini_x,ini_y,fi_x,fi_y,previous_num_whites,zone)

			print 'En aquesta zona hi ha: Motion positius (',positives,')/(',negatives,') i el resultat d warm es: ',w_result


			iterations+=1
			
		
	def warmColorDetection(self,img_np_arr,ini_x,ini_y,fi_x,fi_y,previous_num_whites,zone):
		#self.mutex.acquire()
		
		#self.mutex.release()

		positives = 0
		negatives = 0
		max_num_colors = (fi_x-ini_x)*(fi_y-ini_y)
		
		segment = img_np_arr[ini_x:fi_x, ini_y:fi_y,:]
		freq = None
		
		p = 0
		n = 0
		for x in range(fi_x-ini_x):
			for y in range(fi_y-ini_y):
				R = segment[x,y,0]
				G = segment[x,y,1]
				B = segment[x,y,2]
				bgr = [B,G,R]
				if bgr in colors2:
					p+=1
				else:
					n+=1

		if p >= max_num_colors * self.color_tolerance:
			#self.mutex.acquire()
			
			self.results_color_detection[zone].append(('Zona: '+str(zone),True))
			
			#self.mutex.release()
			return True
		else:
			#self.mutex.acquire()
			self.results_color_detection[zone].append(('Zona: '+str(zone),False))
			
			
			#self.mutex.release()

			return False	

if __name__ == '__main__':
	f = FirefAIghter()	
		