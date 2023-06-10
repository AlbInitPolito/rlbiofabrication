'''
Credits: Alberto Castrignanò, s281689, Politecnico di Torino
'''

import gym
import subprocess
import xml.etree.ElementTree as ET
import tensorflow as tf
import env.Palacell.vtkInterface as vki
import os
from xml.dom import minidom

disc_acts = ["X","Y"]
class PalacellEnv(gym.Env):
  def __init__(self,width=300, height=300, lr=0.001, gamma=0.99,iters=20,max_iterations=4200,
   preload_model_weights=None, preload_model_scores=None, preload_observations=None, output_dir="palacell_out"):
    self.configuration_dir = "."
    self.configuration = self.configuration_dir+"/compr.xml"
    self.iters = iters
    self.epochs = 300
    self.iterations = int(max_iterations/iters)+1
    self.iteration_num = 0
    self.width = width
    self.height = height
    self.channels = 3
    self.lr = lr
    self.gamma = gamma
    self.preload_model_weights = preload_model_weights
    self.preload_model_scores = preload_model_scores
    self.preload_observations = preload_observations
    self.output_dir = output_dir
    self.num_continue = 1
    self.num_discrete = 1
    self.range_continue = [(0,0.1)]
    self.dim_discrete = [2]
    self.least_iterations = max_iterations


  def reset(self): #todo!
    cwd = os.getcwd()
    os.chdir('env/Palacell/PalaCell2D/app')
    super().reset()
    self.configure(self.configuration,0)
    try:
      process = subprocess.Popen(['./palaCell',self.configuration], stdout=subprocess.DEVNULL)
      process.wait()
    except Exception as e:
      os.chdir(cwd)
      raise Exception(e)
    image = vki.create_pil_image("output/chem_1-_final_cell")
    observation = vki.pil_to_array(image).reshape((1,self.width,self.height,3)).copy()
    self.iteration_num = 0
    self.last_cell_num = 1
    os.chdir(cwd)
    return observation

  def _get_info(self):
    st = "last cell num: "+str(self.last_cell_num)+", best least iterations num: "+str(self.least_iterations)
    return st

  def step(self, action):
    cwd = os.getcwd()
    os.chdir('env/Palacell/PalaCell2D/app')
    self.configure(self.configuration, self.iters, action[0], action[1], export_step=self.iters, initialPath = "output/chem_1-_final_cell.vtp")
    try:
      process = subprocess.Popen(['./palaCell',self.configuration], stdout=subprocess.DEVNULL)
      process.wait()
    except Exception as e:
      os.chdir(cwd)
      raise Exception(e)

    os.chdir(cwd)
    observation = tf.keras.preprocessing.image.img_to_array(self.render()).reshape((1,self.width,self.height,3))
    cwd = os.getcwd()
    os.chdir('env/Palacell/PalaCell2D/app')
    cell_num = vki.read_cell_num("output/chem_1-_final_cell")
    reward = cell_num - self.last_cell_num
    self.last_cell_num = cell_num
    self.iteration_num += self.iters
    #done = 1 if self.iteration_num>=self.max_iterations else 0
    done = True if cell_num>=300 else False
    if done:
      if self.iteration_num < self.least_iterations:
        self.least_iterations = self.iteration_num 
    os.chdir(cwd)
    return observation, reward, done, None

  def render(self):
    cwd = os.getcwd()
    os.chdir('env/Palacell/PalaCell2D/app')
    image = vki.create_pil_image("output/chem_1-_final_cell").resize([self.width,self.height])
    os.chdir(cwd)
    return image

  def adapt_actions(self, discrete_actions = None, continue_actions=None):
    return [disc_acts[discrete_actions[0]],str(continue_actions[0].numpy())]

  def configure(self, filePath, num_iter=5, axis='X', compr_force=0.0, export_step='0', init='0', initialPath = ' ', initialWallPath = ' '):
    root = ET.Element('parameters')

    geometry = ET.SubElement(root, 'geometry')
    simulation = ET.SubElement(root, 'simulation')
    physics = ET.SubElement(root, 'physics')
    numerics = ET.SubElement(root, 'numerics')

    #geometry
    initialVTK = ET.SubElement(geometry, 'initialVTK')
    initialWallVTK = ET.SubElement(geometry, 'initialWallVTK')
    finalVTK = ET.SubElement(geometry, 'finalVTK')
    finalVTK.text = "chem_1-"
    initialVTK.text = initialPath
    initialWallVTK.text = initialWallPath

    #simulation
    type = ET.SubElement(simulation,"type")
    exportStep = ET.SubElement(simulation,"exportStep")
    initStep = ET.SubElement(simulation,"initStep")
    verbose = ET.SubElement(simulation,"verbose")
    exportCells = ET.SubElement(simulation,"exportCells")
    exportForces = ET.SubElement(simulation,"exportForces")
    exportField = ET.SubElement(simulation,"exportField")
    exportSpecies = ET.SubElement(simulation,"exportSpecies")
    exportDBG = ET.SubElement(simulation,"exportDBG")
    exportCSV = ET.SubElement(simulation,"exportCSV")
    seed = ET.SubElement(simulation,"seed")
    exit = ET.SubElement(simulation,"exit")
    numIter = ET.SubElement(simulation,"numIter")
    numTime = ET.SubElement(simulation,"numTime")
    numCell = ET.SubElement(simulation,"numCell")
    startAt = ET.SubElement(simulation,"startAt")
    stopAt = ET.SubElement(simulation,"stopAt")

    type.text = '1'
    exportStep.text = str(export_step)
    initStep.text = init
    verbose.text = '0'
    exportCells.text = "true"
    exportForces.text = "false" #da questi possono dipendere le simulazioni successive?
    exportField.text = "false" #intendo anche questo
    exportSpecies.text = "false" #e questo
    exportDBG.text = "false"
    exportCSV.text = "false"
    seed.text = '40' #-1?
    exit.text = "iter"
    numIter.text = str(num_iter)
    numTime.text = '7200' #perchè?
    numCell.text = '300'
    startAt.text = '0'
    stopAt.text = str(num_iter)

    #physics
    diffusivity = ET.SubElement(physics, "diffusivity")
    reactingCell = ET.SubElement(physics, "reactingCell")
    reactionRate = ET.SubElement(physics, "reactionRate")
    dissipationRate = ET.SubElement(physics, "dissipationRate")
    growthThreshold = ET.SubElement(physics, "growthThreshold")
    zeta = ET.SubElement(physics, "zeta")
    rho0 = ET.SubElement(physics, "rho0")
    d0 = ET.SubElement(physics, "d0")
    dmax = ET.SubElement(physics, "dmax")
    n0 = ET.SubElement(physics, "n0")
    numCells = ET.SubElement(physics, "numCells")
    cell = ET.SubElement(physics, "cell")
    numVertex = ET.SubElement(physics, "numVertex")
    edgeVertex = ET.SubElement(physics, "edgeVertex")
    vertex = ET.SubElement(physics, "vertex")
    vertex_1 = ET.SubElement(physics, "vertex")
    vertex_2 = ET.SubElement(physics, "vertex")
    extern = ET.SubElement(physics, "extern")

    diffusivity.text = '2.0'
    reactingCell.text = '0'
    reactionRate.text = '0.001'
    dissipationRate.text = '0.0001'
    growthThreshold.text = '0.025'
    zeta.text = '0.7'
    rho0.text = '1.05'
    d0.text = '0.5'
    dmax.text = '1.0'
    n0.text = '123'
    numCells.text = '1'
    cell.attrib['type'] = 'default'
    numVertex.text = '3'
    edgeVertex.text = '-1'
    vertex.attrib['type'] = 'default'
    vertex_1.attrib['type'] = '1'
    vertex_2.attrib['type'] = '2'

    #physics.cell
    divisionThreshold = ET.SubElement(cell,"divisionThreshold")
    pressureSensitivity = ET.SubElement(cell,"pressureSensitivity")
    nu = ET.SubElement(cell,"nu")
    nuRelax = ET.SubElement(cell,"nuRelax")
    A0 = ET.SubElement(cell,"A0")
    k4 = ET.SubElement(cell,"k4")
    probToProlif = ET.SubElement(cell,"probToProlif")
    maxPressureLevel = ET.SubElement(cell,"maxPressureLevel")
    zetaCC = ET.SubElement(cell, "zetaCC")

    divisionThreshold.text = '300.0'
    pressureSensitivity.text = '2.5'
    nu.text = '0.0025'
    nuRelax.text = '0.01'
    A0.text = '300.0'
    k4.text = '0.01'
    probToProlif.text = '0.001'
    maxPressureLevel.text = '0.05'
    zetaCC.text = '0.4'

    #physics.vertex
    k1 = ET.SubElement(vertex, "k1")
    k3 = ET.SubElement(vertex, "k3")
    k1.text = '0.2'
    k3.text = '0.2'
    k1_1 = ET.SubElement(vertex_1, "k1")
    k1_2 = ET.SubElement(vertex_2, "k3")
    k1_1.text = '0.001'
    k1_2.text = '0.2'

    #physics.extern
    compressionAxis = ET.SubElement(extern, "compressionAxis")
    comprForce = ET.SubElement(extern, "comprForce")
    kExtern = ET.SubElement(extern, "kExtern")
    center = ET.SubElement(extern, "center")
    rMin = ET.SubElement(extern, "rMin")

    compressionAxis.text = axis
    comprForce.text = str(compr_force)
    kExtern.text = '0.01'
    center.text = '200 200'
    rMin.text = '100'

    #numerics
    dx = ET.SubElement(numerics, "dx")
    dt = ET.SubElement(numerics, "dt")
    domain = ET.SubElement(numerics, "domain")
    
    dx.text = '1.0'
    dt.text = '1.0'
    domain.text = '0 0 400. 400.'

    tree = ET.ElementTree(root)
    #ET.indent(root)
    #tree.write(filePath, xml_declaration=True)
    indented_xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(filePath, "w") as f:
        f.write(indented_xml)

if __name__=="__main__":
  env = PalacellEnv()
  obs = env.reset()
  print(obs.shape)
  img = vki.array_to_pil(obs)
  img.show()
  input()
