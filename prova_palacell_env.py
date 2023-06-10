import env.Palacell.PalacellEnv as penv
import env.Palacell.vtkInterface as vki

if __name__== "__main__":
    env = penv.PalacellEnv()
    obs = env.reset()
    print(obs.shape)
    img = vki.array_to_pil(obs)
    img.show()
    input()
    for _ in range(40):
        env.step(['X','0.001'])
    img = env.render()
    img.show()
    