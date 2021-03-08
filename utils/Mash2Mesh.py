'''
Points generated Code for
Integrating 3D Geometry of Organ for Improving Medical Image Segmentation
J Yao, J Cai, D Yang, D Xu, J Huang - MICCAI, 2019

'''

import vtk
# import bpy


class Mask2Mesh():

    def __init__(self, NameOrImage, target=1):

        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(NameOrImage)
        reader.Update()

        threshold = vtk.vtkImageThreshold()
        threshold.SetInputConnection(reader.GetOutputPort())
        threshold.ThresholdBetween(target, target+0.5)  # remove all soft tissue
        threshold.ReplaceInOn()
        threshold.SetInValue(1)  # set all values below 400 to 0
        threshold.ReplaceOutOn()
        threshold.SetOutValue(0)  # set all values above 400 to 1
        threshold.Update()

        self.Image = reader.GetOutput()
        self.points = []
        self.triangles = []
        self.meshes = {}
        self.meshes_resample = {}
        self.threshold = threshold

    def getMesh(self, target=1, spacing=[2, 2, 2]):

        mesh = vtk.vtkMarchingCubes()
        # mesh.SetInputData(self.Image)
        mesh.SetInputConnection(self.threshold.GetOutputPort())
        mesh.GenerateValues(1, 1, 1)
        mesh.Update()

        self.meshes[str(target)] = mesh.GetOutput()

        mesh = self.meshes[str(target)]
        points = [mesh.GetPoint(i) for i in range(mesh.GetPoints().GetNumberOfPoints())]
        mesh.GetPolys().InitTraversal()
        idList = vtk.vtkIdList()
        triangles = []
        while mesh.GetPolys().GetNextCell(idList):
            assert idList.GetNumberOfIds() == 3
            triangles.append([idList.GetId(0), idList.GetId(1), idList.GetId(2)])

        self.points = points
        self.triangles = triangles
