'''
Credits: Alberto Castrignanò, s281689, Politecnico di Torino
'''

import numpy as np
import pygame
import time
import cv2
import os
from os import listdir

'''
reads a numpy file containing an array of images, and generates an mp4 video
'''
def generate_video(filename, width=300, height=300):
    images = np.load(filename+".npy", allow_pickle=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video=cv2.VideoWriter(filename+'.mp4',fourcc,1,(width,height))
    for image in images:
        video.write(np.asarray(image.resize((width,height))))
    cv2.destroyAllWindows()
    video.release()

'''
reads a numpy array containing images, and generates an mp4 video
'''
def generate_video_from_array(images, output_file, width=300, height=300):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video=cv2.VideoWriter(output_file+'.mp4',fourcc,1,(width,height))
    for image in images:
        video.write(np.asarray(image.resize((width,height))))
    cv2.destroyAllWindows()
    video.release()

'''
reads all numpy files in a directory and generates more mp4 videos
'''
def generate_video_from_directory(directory, width=300, height=300):
    files = {directory+"/"+filename: np.load(directory+"/"+filename,allow_pickle=True) for filename in listdir(directory) if filename.split('.')[-1]=='npy'}
    for file in files:
        images = files[file]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video=cv2.VideoWriter(os.path.splitext(file)[0]+'.mp4',fourcc,1,(width,height))
        for image in images:
            video.write(np.asarray(image))
        cv2.destroyAllWindows()
        video.release()

if __name__=='__main__':
    generate_video("out/cartpole_300_001_99/observations")
    '''
    print("on")
    input()
    time.sleep(5)

    screen = pygame.display.set_mode(300,300))
    pygame.display.flip()
    for image in images:
        raw = image.tobytes("raw", "RGB")
        pygame_surface = pygame.image.fromstring(raw, (300,300), "RGB") 
        screen.blit(pygame_surface, (0,0))
        #pygame.display.update()
        pygame.display.flip()
        time.sleep(0.03)

    #while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    '''