import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter as dsa
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkWindowToImageFilter
)
from vtkmodules.vtkIOImage import (
    vtkJPEGWriter,
    vtkPNGWriter,
)

import PIL.Image as Image
import numpy as np
import os
import tensorflow.keras.preprocessing.image as kpi

#cellData = polydata.GetCellData()
#print(cellData.GetPoints())
def read_cell_num(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename+".vtp")
    reader.Update()
    polydata = reader.GetOutput()
    return polydata.GetNumberOfCells()
    #polydata = reader.GetOutput()
    #numpy_array_of_points = dsa.WrapDataObject(polydata).Points
    #cellData = polydata.GetCellData()
    #print(cellData.GetPoints())
    

def create_png_from_vtk(filename):
    colors = vtkNamedColors()
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(filename+".vtp")
    reader.Update()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d('White'))
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.SetOffScreenRendering(1)
    renWin.AddRenderer(ren)
    ren.AddActor(actor)
    ren.SetBackground(colors.GetColor3d('Black'))
    renWin.Render()
    writer = vtkPNGWriter()
    windowto_image_filter = vtkWindowToImageFilter()
    windowto_image_filter.SetInput(renWin)
    windowto_image_filter.SetScale(1)  # image quality
    windowto_image_filter.SetInputBufferTypeToRGB()
    windowto_image_filter.ReadFrontBufferOff()
    windowto_image_filter.Update()
    writer.SetFileName(filename+".png")
    writer.SetInputConnection(windowto_image_filter.GetOutputPort())
    writer.Write()

def create_pil_image(filename,delete=True):
    create_png_from_vtk(filename)
    image = Image.open(filename+".png").convert("RGB")
    if delete:
        if os.path.exists(filename+".png"):
            os.remove(filename+".png")
        else:
            print("Error in creating png image")
            exit(-99)
    return image

def pil_to_array(image):
    return np.asarray(image).copy()

def array_to_pil(array,width=300,height=300):
    return Image.fromarray(np.squeeze(array), 'RGB').copy().resize([width,height])

def keras_obs(image,width=300,height=300,channels=3):
    return kpi.img_to_array(image).reshape((1,width,height,channels))

if __name__ == '__main__':
    #create_image_from_vtk("provavtk")
    image = create_pil_image("provavtk",delete=False)
    print(image)
    array = pil_to_array(image)
    print(array)
    image = array_to_pil(array)
    print(image)
    print(keras_obs(image))
    print(read_cell_num("provavtk"))
    

    
