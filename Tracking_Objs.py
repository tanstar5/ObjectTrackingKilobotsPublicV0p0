import json
import numpy as np
import os
import matplotlib.pyplot as plt



class Tracking_Objs:
    def __init__(self):
        self.ID = np.empty((0,0));
        self.x_pos = np.empty((0,0));
        self.y_pos = np.empty((0,0)); # [ID,X_coor,Y_coor,t_coor] ndmin = 2
        self.frame_no = np.empty((0,0));
        self.loop_interval = np.empty((0,0));
        
        
    def add_obj_info(self,ID,x,y,t,dt):
        self.ID = np.append(self.ID,ID);
        self.x_pos = np.append(self.x_pos,x);
        self.y_pos = np.append(self.y_pos,y);
        self.frame_no = np.append(self.frame_no,t);
        self.loop_interval = np.append(self.loop_interval,dt);

    def get_ObjInfo(self):
        print(f'{self.ID}j')
        
        
    ## Algorithm functions for tracking
    def find_obj_inframe(self,frame_no):
        IDs = self.ID;
        x_pos = self.x_pos;
        y_pos = self.y_pos;
        dt = self.loop_interval;
        t = self.frame_no;
        IDs_interesting = IDs[t == frame_no]; 
        x_pos_interesting = x_pos[t == frame_no];
        y_pos_interesting = y_pos[t == frame_no];
        dt_interesting = dt[t == frame_no];
        t_interesting = t[t == frame_no];
        return IDs_interesting,x_pos_interesting,y_pos_interesting,t_interesting, dt_interesting
        
    def assign_IDs_to_newCoors(self,next_frame_objs_coors,radius,from_frame):
        IDs_interesting,x_pos_interesting,y_pos_interesting,t_interesting, dt_interesting = \
            self.find_obj_inframe(from_frame)
        # print('DEBUG:IDs_interesting',IDs_interesting);
        # print('DEBUG:x_pos_interesting',x_pos_interesting);
        # print('DEBUG:y_pos_interesting',y_pos_interesting);
        # print('DEBUG:t_interesting',t_interesting);
        # print('DEBUG:dt_interesting',dt_interesting);
        prev_frame_obj_connected = np.zeros( (len(IDs_interesting)) );
        index_prev_obj = np.arange(0,len(IDs_interesting),1);
        next_frame_IDs = np.empty(( len(next_frame_objs_coors[:,0]) ));
        #print('DEBUG:next_frame_IDs.shape',next_frame_IDs.shape);
        updated_coors = next_frame_objs_coors;
        index_new_obj = np.arange(0,len(next_frame_objs_coors[:,0]),1);
        for ind in index_new_obj:
            coor_x = next_frame_objs_coors[ind,0];
            coor_y = next_frame_objs_coors[ind,1];
            #print('DEBUG: Calculation distance for:',ind);
            distance = ((x_pos_interesting-coor_x)**2+(y_pos_interesting-coor_y)**2)**(0.5);
            #print('DEBUG:distances',distance);
            distance_minimum = np.min(distance);
            #print('DEBUG:distance_minimum',distance_minimum);
            if(distance_minimum<=radius):
                #print('DEBUG len(IDs_interesting[distance==distance_minimum])',len(IDs_interesting[distance==distance_minimum]))
                next_frame_IDs[ind] = IDs_interesting[distance==distance_minimum][0];
                prev_frame_obj_connected[distance==distance_minimum] = 1;
            else:
                next_frame_IDs[ind] = float('nan');
        #print('DEBUG:Assigned IDs',next_frame_IDs);
        prev_frame_IDs_not_connected = IDs_interesting[prev_frame_obj_connected==0];
        #print('DEBUG: prev_frame_IDs_not_connected:',prev_frame_IDs_not_connected );
        for prev_id in prev_frame_IDs_not_connected:
            next_frame_IDs = np.append(next_frame_IDs,prev_id);
            corresponding_coor = \
                np.array([ x_pos_interesting[IDs_interesting==prev_id],y_pos_interesting[IDs_interesting==prev_id] ]) ;            
            #print('DEBUG: corresponding_coor', corresponding_coor[:,0])
            updated_coors = np.append(updated_coors,[corresponding_coor[:,0]],axis = 0);
        
        #print('Asigning new ids to the new objects');
        index = np.arange(0,len(next_frame_IDs),1);
        for inds in index:
            if(next_frame_IDs[inds]!=next_frame_IDs[inds]):
                #print('DEBUG: nan id detected')
                next_frame_IDs[inds] = np.nanmax(next_frame_IDs)+1;
        #print('DEBUG: RESULT next_frame_IDs,updated_coors;',next_frame_IDs,updated_coors)        
        return next_frame_IDs, updated_coors
            
    
    def detect_me_next_frame_and_update(self,next_frame_objs_coors, radius,frame,dt):
        #filter out coors with same Center of mass or close less than size of the objects
        next_frame_objs_coors,_ = filter_duplicate_objs(next_frame_objs_coors)        
        #Check initial condition if any particles detected before
        ID_mes = self.ID;
        newID_list = np.empty((0,0));
        new_coor_list = np.empty((0,2),int);
        if (len(ID_mes) == 0 ):            
            print('No previous object found. Adding new objects')
            index = np.arange(0,len(next_frame_objs_coors[:,0]),1);
            for ind in index:
                currID = ind;
                curr_X = next_frame_objs_coors[ind,0];                
                curr_Y = next_frame_objs_coors[ind,1]; 
                corresponding_coor = [next_frame_objs_coors[ind,:]]
                new_coor_list = np.append(new_coor_list,corresponding_coor,axis=0);
                newID_list = np.append(newID_list,currID);
                self.add_obj_info(currID,curr_X,curr_Y,frame,dt);
        else:
            print('Previous objects found and so linking');
            prev_frame_info = self.frame_no;
            prev_x_info = self.x_pos;
            prev_y_info = self.y_pos;
            prev_ID = self.ID;
            prev_loop_interval = self.loop_interval;

            #Last frame info only
            last_frame = np.max(prev_frame_info);
            next_frame_IDs, updated_coors = \
                self.assign_IDs_to_newCoors(next_frame_objs_coors,radius,last_frame)
            index = np.arange(0,len(next_frame_IDs),1);
            for ind in index:
                curr_X = updated_coors[ind,0];
                curr_Y = updated_coors[ind,1];
                self.add_obj_info(next_frame_IDs[ind],curr_X,curr_Y,frame,dt);
                corresponding_coor = [updated_coors[ind,:]]
                new_coor_list = np.append(new_coor_list,corresponding_coor,axis=0);
                newID_list = np.append(newID_list,next_frame_IDs[ind]);
            new_coor_list,index_not_to_del  = filter_duplicate_objs(new_coor_list)             
            newID_list = newID_list[index_not_to_del];
        return newID_list,  new_coor_list      
    
    def arange_objs_according_to_IDs(self,num_analysis_frames_from_present=100):
        '''Returns list of dicts of objects with their properties as key value pairs '''
        all_frames = self.frame_no;
        curr_frame = np.max(all_frames);
        frames_to_consider_logical = all_frames>=(curr_frame-num_analysis_frames_from_present);
        intersting_ID_list = self.ID[frames_to_consider_logical];
        intersting_x_pos_list = self.x_pos[frames_to_consider_logical];
        intersting_y_pos_list = self.y_pos[frames_to_consider_logical];
        frame_no_list = self.frame_no[frames_to_consider_logical];
        loop_interval_list = self.loop_interval[frames_to_consider_logical];
        unique_intersting_ID_list = np.unique(intersting_ID_list);
        list_of_objs = [];
        for unq_id in unique_intersting_ID_list:
            x_pos_current_id = intersting_x_pos_list[intersting_ID_list==unq_id];
            y_pos_current_id = intersting_y_pos_list[intersting_ID_list==unq_id];
            frame_no_current_id = frame_no_list[intersting_ID_list==unq_id];            
            loop_interval_current_id = loop_interval_list[intersting_ID_list==unq_id];
            curr_obj =\
                dict([('ID',unq_id),('x_pos',x_pos_current_id),('y_pos',y_pos_current_id),('frame_no',frame_no_current_id),('loop_interval',loop_interval_current_id)]);
            list_of_objs.append(curr_obj);
        return list_of_objs;
    
    def kill_static_objects(self,live_time):
        list_of_objs = self.arange_objs_according_to_IDs(live_time);
        #area,displacement, distance = self.track_analysis(live_time);
        #print('DEBUG list_of_objs',list_of_objs[1])
        #print('DEBUG distance',distance)
        dead_objs = list_of_objs;
        for dd_objs in dead_objs:
            area, displacement,distance = calculate_trace_parameters(dd_objs);
            if distance !=0:
                dead_objs.remove(dd_objs);
            

                
        ID = np.empty((0,0));
        for dd_ob in dead_objs:
            ID = np.append(ID,dd_ob["ID"]);
        print('DEBUG ID', ID) 
        print('DEBUG self.ID',self.ID)
        self.ID = self.ID[self.ID!=ID];
        self.x_pos = self.x_pos[self.ID!=ID];
        self.y_pos = self.y_pos[self.ID!=ID];
        self.frame_no = self.frame_no[self.ID!=ID];
        self.loop_interval = self.loop_interval[self.ID!=ID]
        
        

    ## Analysis
    def track_analysis(self,num_analysis_frames_from_present=100):
        '''generates plot of neighbours vs velocity histogram analysis'''
        list_of_objs = self.arange_objs_according_to_IDs(num_analysis_frames_from_present=100);
        area = np.empty((0,0));
        displacement = np.empty((0,0));
        distance = np.empty((0,0));
        for curr_obj in list_of_objs:
            area1,displacement1,distance1 = calculate_trace_parameters(curr_obj)
            #print(area)
            area = np.append(area,area1);
            displacement = np.append(displacement,displacement1 );
            distance = np.append(distance,distance1 );
        return area,displacement, distance  

        
      
    ## File and memory management
    def write_coors_tofile_and_dump_frames(self,recent_frame_info_to_keep,write_to_file=1,fname='output.json'):
        frame_values = self.frame_no;
        if (len(frame_values)==0):
            print('MESSAGE:Nothing to delete');
        else: 
            print('MESSAGE:Looking for frames to delete');
            recent_frame = np.max(frame_values);
            from_frames_to_del = recent_frame-recent_frame_info_to_keep;
            if (from_frames_to_del>0):
                print('MESSAGE: Deleting frames from: from_frames_to_del',from_frames_to_del);
                index = np.arange(0,len(frame_values),1);
                ind_to_del = index[frame_values<from_frames_to_del];
                if ( len(ind_to_del)==0 ):
                    print('MESSAGE: Yet not enough frames to delete:len(ind_to_del)',len(ind_to_del));
                else:
                    if(write_to_file==1):
                        print('MESSAGE:Writing to file');
                        entry = Tracking_Objs();
                        entry.ID = self.ID[ind_to_del].tolist();
                        entry.x_pos = self.x_pos[ind_to_del].tolist();
                        entry.y_pos = self.y_pos[ind_to_del].tolist();
                        entry.frame_no = self.frame_no[ind_to_del].tolist();
                        entry.loop_interval = self.loop_interval[ind_to_del].tolist();
                        entry = entry.__dict__
                        a = []
                        if not os.path.isfile(fname):
                            #a.append(entry)
                            with open(fname, mode='w') as f:
                                f.write(json.dumps(entry, indent=1))
                        else:
                            # with open(fname) as feedsjson:
                            #     feeds = json.load(feedsjson)
                        
                            #feeds.append(entry)
                            with open(fname, mode='a') as f:
                                f.write(json.dumps(entry, indent=1))
                                
                    self.ID = np.delete(self.ID, ind_to_del);
                    self.x_pos = np.delete(self.x_pos, ind_to_del);
                    self.y_pos = np.delete(self.y_pos, ind_to_del);
                    self.frame_no = np.delete(self.frame_no, ind_to_del);
                    self.loop_interval = np.delete(self.loop_interval, ind_to_del);
                    
            else:
                print('MESSAGE: Not enough frame to delete')
                
            
        
#other accesory functions
def calculate_trace_parameters(curr_obj):
    x_pos = curr_obj["x_pos"];
    y_pos = curr_obj["y_pos"];
    num_coors = len(x_pos);
    displacement = ((x_pos[0] - x_pos[num_coors-1])**2 + (y_pos[0] - y_pos[num_coors-1])**2 )**(.5)
    distance = np.sum(((x_pos[1:num_coors] - x_pos[0:num_coors-1])**2 + (y_pos[1:num_coors] - y_pos[0:num_coors-1])**2)**(0.5) );
    
    area = 0;    
    if(num_coors>2):
        for i in range(1,num_coors-1):
            vec1 = np.array([x_pos[0] -x_pos[i] , y_pos[0] -y_pos[i]  ]);
            vec2 = np.array([x_pos[0] -x_pos[i+1] , y_pos[0] -y_pos[i+1]  ]);
            area = area + np.cross(vec1,vec2);
        
    else:
        area = 0;
    
    return area, displacement,distance
            
        
def plot_analysis_results(area, displacement,distance,fig, axes):
   # write %matplotlib qt in console to plot in new window
   # generate area displacement phase scattter plot
   area_filtered = area[distance>0];
   displacement_filterred = displacement[distance>0];
   axes[0][0].scatter(displacement_filterred,area_filtered);
   
    
                
def filter_duplicate_objs(centers_current,duplicate_rad=5):
    index = np.arange(0,len(centers_current[:,0]),1);
    noter = np.zeros(len(index));
    noter_modified = np.zeros(len(index));
    for ind in index:
        curr_X = centers_current[ind,0];                
        curr_Y = centers_current[ind,1]; 
        distance_all = ((centers_current[:,0]-curr_X) **2 +  (centers_current[:,1]-curr_Y) **2)**(0.5)
        
        if(len( distance_all[distance_all<=duplicate_rad] )>1 and noter_modified[ind]==0  ):
            noter[ind] = 1;
            noter_modified[distance_all<=duplicate_rad] = 1; 
    index_to_delete = index[noter==1];
    index_not_to_del = index[noter!=1];
    centers_current = centers_current[index_not_to_del,:]; 
    return centers_current,index_not_to_del;



tracking_objs_test =  Tracking_Objs();
#tracking_objs_test.add_obj_info(1,0,0,3,1)        
next_frame_objs_coors = np.array([[1,4],[2,1]])
tracking_objs_test.detect_me_next_frame_and_update(next_frame_objs_coors,100,0,1)
next_frame_objs_coors = np.array([[1,4],[2,1],[3,6]])
tracking_objs_test.detect_me_next_frame_and_update(next_frame_objs_coors,1,1,1)
next_frame_objs_coors = np.array([[1,4],[2,1],[3,8]])
tracking_objs_test.detect_me_next_frame_and_update(next_frame_objs_coors,1,2,1)
#tracking_objs_test.write_coors_tofile_and_dump_frames(1,1,"output3.json")
tracking_objs_test.kill_static_objects(2)

