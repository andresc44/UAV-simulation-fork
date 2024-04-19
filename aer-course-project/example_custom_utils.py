"""Example utility module.

Please use a file like this one to add extra functions.

"""
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import itertools
show_animation = True

def exampleFunction():
    """Example of user-defined function.

    """
    x = -1
    return x


class AStar():
    def __init__(self, config=None):
        # super(AStar, self).__init__(config)
        # self.resolution = 0.1
        # self.rr = 0.1
        self.resolution = 10
        self.rr = 10
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()

        self.x_center=[]
        self.y_center=[]
        self.radius=[]

        

    
    
    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)
        
    def plan(self, start, goal, obstacles: list,show_animation=True,mapbound=[-55,-25,55,25] ) -> list:  #[-55,-25,55,25]
        """ Plan a path from start to goal avoiding obstacles.

        Args:
            start (list): list of x, y coordinates of the start point
            goal (list): list of x, y coordinates of the goal point
            obstacles (list): list of obstacles, each obstacle is a tuple (x, y, 0)
            radius (list): list of radius of obstacles
        
        Returns:
            list: This must be a valid list of connected nodes that form
                a path from start to goal node
        """
        
        # obstacles=np.array(obstacles)
        # print("obs:",obstacles)
       
        self.x_center=obstacles[:,0]
        self.y_center=obstacles[:,1]
        self.raius=obstacles[:,2]
        obs_points=[]
        round_obs=np.array([self.x_center,self.y_center,self.raius]).T
        for x_cent, y_cent, rad in obstacles:
            # points = circle_to_points(x_center, y_center, radius, num_points=16)
            points = self.circle_to_grid_cells(x_cent, y_cent, rad)
            obs_points.extend(points)
        ox,oy=[],[]
        self.min_x, self.min_y, self.max_x, self.max_y = mapbound
        # print("number of obs!!!!!!!!!!!:",obstacles.shape)
        # print(self.min_x)
        wall_res=10
        for i in range(self.min_x,self.max_x,wall_res):
            ox.append(i)
            oy.append(self.min_y)
        for i in range(self.min_y,self.max_y,wall_res):
            ox.append(self.max_x)
            oy.append(i)
        for i in range(self.min_x,self.max_x,wall_res):
            ox.append(i)
            oy.append(self.max_y)
        for i in range(self.min_y,self.max_y,wall_res):
            ox.append(self.min_x)
            oy.append(i)

        for i in obs_points:
            ox.append(i[0])
            oy.append(i[1])

        self.calc_obstacle_map(ox, oy)
        

        

        # sx,sy=start

        if isinstance(start,np.ndarray):

            sx,sy=start
        else:
            sx=start.x
            sy=start.y
        # print("goal:",goal)
        # gx=goal.x
        # gy=goal.y
        if isinstance(goal,np.ndarray):

            gx,gy=goal  
        else:
            gx=goal.x
            gy=goal.y


        if show_animation:  # pragma: no cover
            plt.clf()
            # plt.gca().invert_yaxis()
            plt.grid(True)
            plt.scatter(ox, oy,s=2)
            plt.plot(sx, sy, "og")
            plt.plot(gx, gy, "xb")
            
            plt.axis("equal")
            # plt.show(block=False)
            

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        
        
        
        reversed_path=np.array([rx,ry,np.ones(len(rx))]).T
        # print(reversed_path)
        path=reversed_path[::-1]
        extended_path= self.add_orientation_to_path(path)
        # print(extended_path)


        if show_animation:  # pragma: no cover
        #     print("showwww")
        #     plt.plot(rx, ry, "-r")
        #     plt.pause(0.001)



            # plt.clf()
            # plt.pause(1)
            # print(rx)
            # print(ry)
            # plt.show()
            self.plot_path_with_orientations(extended_path)
        
        
        
        return extended_path
    
    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry
    def plot_path_with_orientations(self, path_with_orientation):
        # fig, ax = plt.subplots()
        for i, (x, y, _, zx, zy, zz, zw) in enumerate(path_with_orientation):
            angle = 2 * np.arctan2(zz, zw)  # Convert quaternion back to angle
            # print(angle)
            plt.quiver(x, y, np.cos(angle), np.sin(angle), color= 'red', scale=50,zorder=5)
        
    def add_orientation_to_path(self, path):
    # Function to calculate quaternion from an angle theta
        def angle_to_quaternion(theta):
            return np.array([0, 0, np.sin(theta / 2), np.cos(theta / 2)])
        
        # Initialize the path with orientation
        path_with_orientation = []
        
        # Iterate over the path to calculate the orientation at each step
        for i in range(len(path) - 1):
            x1, y1, _ = path[i]
            x2, y2, _ = path[i + 1]
            
            # Calculate the angle theta between successive points
            theta = np.arctan2(y2 - y1, x2 - x1)
            # print(theta)
            
            # Compute quaternion
            quaternion = angle_to_quaternion(theta)
            
            # Append position and orientation to the new path
            to_append=(x1, y1, 1) + tuple(quaternion)
            
            path_with_orientation.append(to_append)
            
        
        # Add the last point with the same orientation as the second last point
        if path.any():
            # print(path)
            to_append=tuple(path[-1]) + tuple(path_with_orientation[-1][3:])
    
            path_with_orientation.append(to_append)
        path_with_orientation=np.array(path_with_orientation)
        return path_with_orientation
    
    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

        
        
    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
      

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        # print("x_width:", self.x_width)
        # print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break
    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion
    
    def circle_to_grid_cells(self,x_center, y_center, radius, cell_size=1):
        """
        Calculate grid cells occupied by a circular obstacle for any cell size.

        Parameters:
        - x_center, y_center: coordinates of the circle's center in meters.
        - radius: radius of the circle in meters.
        - cell_size: size of each grid cell in meters.

        Returns:
        - A list of (x, y) tuples representing occupied grid cells.
        """
        occupied_cells = []

        # Convert the center position from meters to cell indices
        center_x_index = int(x_center / cell_size)
        center_y_index = int(y_center / cell_size)

        # Convert the radius to cell units
        radius_in_cells = int(radius / cell_size)

        # Calculate the bounding box in cell coordinates
        x_min = center_x_index - radius_in_cells
        x_max = center_x_index + radius_in_cells
        y_min = center_y_index - radius_in_cells
        y_max = center_y_index + radius_in_cells

        # Check each cell in the bounding box to see if it's within the circle
        for x in range(x_min, x_max + 1,2):
            for y in range(y_min, y_max + 1,2):
                # Calculate the cell's center 
                cell_center_x = (x + 0.5) * cell_size
                cell_center_y = (y + 0.5) * cell_size

                # Check if the cell center is inside the circle
                dist=((cell_center_x - x_center) ** 2 + (cell_center_y - y_center) ** 2)
                rad_square=radius ** 2
                if dist <= radius ** 2 and dist >= (radius-2) ** 2:
                    occupied_cells.append((x, y))

        return occupied_cells
    
    # def circle_to_grid_cells(self, x_center, y_center, radius, resolution=1):
    #     """
    #     Calculate grid cells occupied by a circular obstacle.

    #     Parameters:
    #     - x_center, y_center: coordinates of the circle's center in meters.
    #     - radius: radius of the circle in meters.
    #     - resolution: size of each grid cell in meters.

    #     Returns:
    #     - A list of (x, y) tuples representing occupied grid cells.
    #     """
    #     occupied_cells = []

    #     # Calculate the bounding box of the circle
    #     x_min = int((x_center - radius) // resolution)
    #     x_max = int((x_center + radius) // resolution)
    #     y_min = int((y_center - radius) // resolution)
    #     y_max = int((y_center + radius) // resolution)

    #     # Iterate over each cell in the bounding box
    #     for x in range(x_min, x_max ):
    #         for y in range(y_min, y_max ):
    #             # Calculate the center of the cell
    #             cell_center_x = (x ) * resolution
    #             cell_center_y = (y )  * resolution

    #             # Check if the center of the cell is inside the circle
    #             distance = ((cell_center_x - x_center) ** 2 + (cell_center_y - y_center) ** 2) ** 0.5
    #             if distance <= radius:
    #                 occupied_cells.append((x, y))

    #     return occupied_cells                                                           



def main():
    # orders=np.array(list(itertools.permutations([1,2,3,4])))
    orders=[[1,2,3,4]]
    orders=[[2,1,3,4]]
    # orders=[[1,1,2,2,1]]
    # orders=[[2,1,1,3,1]]
    # orders=[[4,4,3,2,1]]
    for order in orders:
        compute_fullpath(order)
    pass

def compute_fullpath(order):
    show_animation = True
    gates=np.array([  # x, y, z, r, p, y, type 
                    [ 0.5, -2.5, 0, 0, 0, -1.57, 0],      # gate 1
                    [ 2.0, -1.5, 0, 0, 0, 0,     0],      # gate 2
                    [ 0.0,  0.2, 0, 0, 0, 1.57,  0],      # gate 3
                    [-0.5,  1.5, 0, 0, 0, 0,     0]])       # gate 4])
    gate_rad=0.07
    obstacles=np.array([  # x, y, z, r, p, y
                        [ 1.5, -2.5, 0, 0, 0, 0],             # obstacle 1
                        [ 0.5, -1.0, 0, 0, 0, 0],             # obstacle 2
                        [ 1.5,    0, 0, 0, 0, 0],             # obstacle 3
                        [-1.0,    0, 0, 0, 0, 0]])              # obstacle 4
    
    # file_name=f"Path_files/path_{order[0]}{order[1]}{order[2]}{order[3]}/"

    file_name=""
   
    for o in order:
        file_name+=f"{o}"
    file_name="Path_files/path_"+file_name + "/"

    if not os.path.isdir(file_name):
        os.makedirs(file_name)
    

    obs=[]
    obs_buffer=0.3
    # obs_buffer=0.25
    for gate in gates:
        if gate[5]!=0: # for vertical gate
            obs.append([gate[0],gate[1]+0.2,gate_rad]) # gate obstacle location
            obs.append([gate[0],gate[1]-0.2,gate_rad])

            obs.append([gate[0]+obs_buffer,gate[1]+0.2,gate_rad]) # buffer obstacle 
            obs.append([gate[0]+obs_buffer,gate[1]-0.2,gate_rad])

            obs.append([gate[0]-obs_buffer,gate[1]+0.2,gate_rad])
            obs.append([gate[0]-obs_buffer,gate[1]-0.2,gate_rad])

            obs.append([gate[0]+obs_buffer,gate[1]+0.5,gate_rad]) #+0.4 to make it wider
            obs.append([gate[0]+obs_buffer,gate[1]-0.5,gate_rad])

            obs.append([gate[0]-obs_buffer,gate[1]+0.5,gate_rad])
            obs.append([gate[0]-obs_buffer,gate[1]-0.5,gate_rad])
        else: #for horizontal gate
            obs.append([gate[0]+0.2,gate[1],gate_rad])  # gate obstacle location
            obs.append([gate[0]-0.2,gate[1],gate_rad])

            obs.append([gate[0]+0.2,gate[1]+obs_buffer,gate_rad]) # buffer obstacle 
            obs.append([gate[0]-0.2,gate[1]+obs_buffer,gate_rad])

            obs.append([gate[0]+0.2,gate[1]-obs_buffer,gate_rad])
            obs.append([gate[0]-0.2,gate[1]-obs_buffer,gate_rad])

            obs.append([gate[0]+0.5,gate[1]+obs_buffer,gate_rad])
            obs.append([gate[0]-0.5,gate[1]+obs_buffer,gate_rad])

            obs.append([gate[0]+0.5,gate[1]-obs_buffer,gate_rad])
            obs.append([gate[0]-0.5,gate[1]-obs_buffer,gate_rad])
    obstacles_rad=0.06
    obstacles_rad=0.2
    obstacles_rad=0.2
    obstacles_noise=0.2

    obs_rad_corrupted=obstacles_rad+obstacles_noise

    for obstacle in obstacles:
        obs.append([obstacle[0],obstacle[1],obs_rad_corrupted])
    obs=np.array(obs)
    obs=obs*100
    
    

    # obs=np.array([#[50,50,20],
    #               #[500,250,30],
    #               #[30,30,10],
    #               [0,100,10]])
    
    mapbound=[-200,-350,300,350] # min_x min_y max_x max_y

    
    planner=AStar()
    start=np.array([-100,-300])
    final_goal=np.array([-50, 200])
    gate4_remove_idx=[]
    for i,value in enumerate(order):
        
        value-=1
        temp_goal=gates[value,0:2]*100
        if i ==0:
            temp_start=start
        else:
            temp_start=path[-1,0:2]*100
        print("iteration::::::::::::::::::::::",i+1)
        print("temp_start:",temp_start)
        print("temp_goal:",temp_goal)

        # temp_goal=np.array([200, -150])

        path=planner.plan(temp_start, temp_goal, obs, show_animation,mapbound)


        direction = np.array([path[-1,0]-path[-2,0], path[-1,1]-path[-2,1]])
        norm = np.linalg.norm(direction)
        direction_unit = direction / norm

        last_pose=path[-1,0:2] ############################change shootout to 10 
        path=np.vstack((path,
                        [last_pose[0]+ direction_unit[0] * 10,last_pose[1]+direction_unit[1] * 10,1,0,0,0,0],
                        [last_pose[0]+ direction_unit[0] * 20,last_pose[1]+direction_unit[1] * 20,1,0,0,0,0],
                        # [last_pose[0]+ direction_unit[0] * 30,last_pose[1]+direction_unit[1] * 30,1,0,0,0,0]
                        ))


        rx=path[:,0]*1
        ry=path[:,1]*1
        # print("unscaled path:",path[:,0:3])

        path=path[:,0:3]
        path[:,:2]=path[:,:2]/100

        
        # print("Path:",path)
        # path.tofile('test_path.dat')
        np.save(file_name+f'test_path{i}.npy', path)    # .npy extension is added if not given


        # obs=np.vstack((obs,[gates[value,0]*100,gates[value,1]*100,2]))            #  obs.append([gate[value,0]*100,gate[value,1]*100,20])

        obs=np.vstack((obs,[gates[value,0]*100+direction_unit[0]*15,gates[value,1]*100+direction_unit[1]*15,2]))        # appending gate blocker    #  obs.append([gate[value,0]*100,gate[value,1]*100,20])

        
        if value==3:
            gate4_remove_idx.append(obs.shape[0])
            print("gate 4 blocker index:",obs.shape[0] )
           

        if show_animation:  # pragma: no cover
            plt.plot(rx, ry, "-r")
            plt.pause(0.001)
            # print(rx)
            # print(ry)
            plt.savefig(file_name+f'test_path{i}.png',bbox_inches='tight',dpi=400)
            plt.show(block=False)

   
    correct_gate4_remove_idx=[idx-1 for idx in gate4_remove_idx]
    print("corrected gate 4 blocker index:",correct_gate4_remove_idx)
    obs=np.delete(obs, correct_gate4_remove_idx, axis=0)

    print("iteration:::::::::::::::::::::: final")
    print("final_start:",path[-1,0:2]*100)
    print("final_goal:",final_goal)
    path=planner.plan(path[-1,0:2]*100, final_goal, obs, show_animation,mapbound)


    # direction = np.array([path[-1,0]-path[-2,0], path[-1,1]-path[-2,1]])
    # norm = np.linalg.norm(direction)
    # direction_unit = direction / norm
    # next_pose=path[-1,0:2]+direction_unit * 30
    # path=np.vstack((path,[next_pose[0],next_pose[1],1,0,0,0,0]))


    rx=path[:,0]*1
    ry=path[:,1]*1
    # print("unscaled path:",path[:,0:3])

    path=path[:,0:3]
    path[:,:2]=path[:,:2]/100

    
    # print("Path:",path)
    # path.tofile('test_path.dat')
    np.save(file_name+'test_path_final.npy', path)    


   

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        # print(rx)
        # print(ry)
        plt.savefig(file_name+'test_path_final.png',bbox_inches='tight',dpi=400)
        plt.show(block=False)

if __name__ == '__main__':
    main()