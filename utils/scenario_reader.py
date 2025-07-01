import numpy as np
import json
import zipfile
import os
import shutil
from matplotlib import pyplot


def get_scene_track(file_path: str) -> tuple:
    """
    Method extracting the track bounds from a scene file.

    :param file_path:    string holding path to a Scenario-Architect archive ('*.saa') or a scene data file ('*.scn')
    :returns (bound_l,   coordinates of the tracks bounds left
              bound_r)   coordinates of the tracks bounds right
    """

    if not (".saa" in file_path or ".scn" in file_path):
        raise ValueError("Unsupported file! Make sure to provide a Scenario-Architect archive ('*.saa') or a scene data"
                         " file ('*.scn').")

    # if archive, extract relevant file
    if ".saa" in file_path:
        zip_obj = zipfile.ZipFile(file_path)
        tmp_file = next((x for x in zip_obj.namelist() if '.scn' in x), None)

        if tmp_file is not None:
            f = zip_obj.open(tmp_file)
            zip_obj.close()
        else:
            raise ValueError("Could not find *.scn file in the provided Scenario-Architect archive!")

    else:
        # directly from file
        f = open(file_path, 'rb')

    # -- read relevant lines from file ---------------------------------------------------------------------------------
    bound_l = None
    bound_r = None
    while True:
        line = f.readline().decode()
        if 'bound_l' in line:
            line = line.replace("# bound_l:", "")
            bound_l = np.array(json.loads(line))
        elif 'bound_r' in line:
            line = line.replace("# bound_r:", "")
            bound_r = np.array(json.loads(line))
        else:
            break

    f.close()

    if bound_l is None or bound_r is None:
        raise ValueError("Something went wrong, while extracting the bound data from the provided file! Check if the"
                         "first two lines of the file hold boundary information.")

    return bound_l, bound_r

def get_scene_ego_traj(file_path: str,
                       append_plan: bool = True,
                       rtrn_safety: bool = False) -> tuple:
    """
    Method extracting the ego-trajectory for a given scenario file (whole duration).

    :param file_path:   string holding the path to the scene data file ('*.scn') or Scenario-Architect archive ('*.saa')
    :param append_plan: if 'True': return not only passed poses, but also append planned ego-traj. from last time-stamp
    :param rtrn_safety: if 'True': return a safety rating for each time-stamps, if present in provided file else "None"
    :returns (time,     time stamps along the trajectory
              x,        x-coordinates along the time-stamps of the ego vehicle
              y,        y-coordinates along the time-stamps of the ego vehicle
              heading,  heading of the ego vehicle along the time-stamps
              curv,     curvature of the path at the position of each time-stamp
              vel,      velocity of the ego vehicle at the position of each time-stamp
              acc)      acceleration of the ego vehicle at the position of each time-stamp
              sfty_dyn, (if enabled) safety with respect to other dynamic vehicles for each t [True, False, None]
              sfty_stat)(if enabled) safety with respect to a static environment for each t [True, False, None]
    """

    if not (".saa" in file_path or ".scn" in file_path):
        raise ValueError("Unsupported file! Make sure to provide a Scenario-Architect archive ('*.saa') or a scene data"
                         " file ('*.scn').")

    # if archive, extract relevant file
    if ".saa" in file_path:
        with zipfile.ZipFile(file_path) as zipObj:
            tmp_file = next((x for x in zipObj.namelist() if '.scn' in x), None)

            if tmp_file is not None:
                with zipObj.open(tmp_file) as zf, open(file_path.replace('.saa', '.scn'), 'wb') as f:
                    shutil.copyfileobj(zf, f)
            else:
                raise ValueError("Could not find *.scn file in the provided Scenario-Architect archive!")

    # retrieve data from file
    data = np.genfromtxt(fname=file_path.replace('.saa', '.scn'),
                         delimiter=";",
                         names=True,
                         skip_header=2,
                         usecols=(0, 1, 2, 3, 4, 5, 6))

    if append_plan:
        # get ego-trajectory from last time stamp (since 'x' and 'y' only holds last poses)
        with open(file_path.replace('.saa', '.scn')) as file:
            # get to top of file (1st line)
            file.seek(0)

            file.readline()
            file.readline()
            header = file.readline()[:-1]

            # extract last line
            line = ""
            for line in file:
                pass

            # parse the data objects we want to retrieve from that line
            ego_traj = np.array(json.loads(dict(zip(header.split(";"), line.split(";")))['ego_traj']))

        # calculate distances along ego-trajectory (in order to determine time-stamps)
        distances = np.sqrt(np.sum(np.power(np.diff(ego_traj[:, 0:2], axis=0), 2), axis=1))

        # calculate time-stamps for ego-trajectory
        t = np.concatenate(([0], np.cumsum(np.divide(distances, ego_traj[:-1, 4],
                                                     out=np.full(ego_traj[:-1, 4].shape[0], np.inf),
                                                     where=ego_traj[:-1, 5] != 0))))

        # fuse file data and last trajectory information
        time = np.concatenate((data['time'], t[1:] + data['time'][-1]))
        x = np.concatenate((data['x'], ego_traj[1:, 0]))
        y = np.concatenate((data['y'], ego_traj[1:, 1]))
        heading = np.concatenate((data['heading'], ego_traj[1:, 2]))
        curv = np.concatenate((data['curv'], ego_traj[1:, 3]))
        vel = np.concatenate((data['vel'], ego_traj[1:, 4]))
        acc = np.concatenate((data['acc'], ego_traj[1:, 5]))

    else:
        time = data['time']
        x = data['x']
        y = data['y']
        heading = data['heading']
        curv = data['curv']
        vel = data['vel']
        acc = data['acc']

    # extract safety data (if requested)
    safety_dyn = None
    safety_stat = None
    if rtrn_safety:
        with open(file_path.replace('.saa', '.scn')) as file:
            # get to top of file (1st line)
            file.seek(0)
            file.readline()
            file.readline()
            header = file.readline()[:-1]

            if "safety_dyn" in header and "safety_stat" in header:
                safety_dyn = [None] * len(time)
                safety_stat = [None] * len(time)

                # extract last line
                for i, line in enumerate(file):
                    safety_dyn[i] = json.loads(line.split(";")[header.split(";").index("safety_dyn")])
                    safety_stat[i] = json.loads(line.split(";")[header.split(";").index("safety_stat")])

    # if it was in archive, remove extracted file after import
    if ".saa" in file_path:
        os.remove(file_path.replace('.saa', '.scn'))

    if not rtrn_safety:
        return time, x, y, heading, curv, vel, acc
    else:
        return time, x, y, heading, curv, vel, acc, safety_dyn, safety_stat
    
def get_scene_timesample(file_path: str,
                         t_in: float,
                         time_f: np.ndarray = None,
                         append_safety: bool = False) -> tuple:
    """
    Method extracting scenario data for a given time instance. If the given time step is not present in the data file,
    linear interpolation is used to generate the desired instance between the neighboring time instances.

    :param file_path:       string holding path to a Scenario-Architect archive ('*.saa') or a scene data file ('*.scn')
    :param t_in:            two options:
                            * float number holding the time-stamp to be extracted [linear interp. between neighb. pts]
                            * int number holding the number of the data reading to be extracted from the file
    :param time_f:          time-stamps from file (for faster exec.: load once and hand to function on next iter)
    :param append_safety:   append safety ground truth in returned tuple if provided in loaded scenario
    :returns (time,         time stamp of the returned sample
              pos,          position of the ego vehicle (list holding x and y)
              heading,      heading of the ego vehicle (in the global frame)
              curv,         curvature of the path at the position of the ego vehicle
              vel,          velocity of the ego vehicle (in the direction of the heading)
              acc,          acceleration of the ego vehicle (in the direction of the heading)
              ego_traj,     planned ego trajectory starting at the current position (x, y, heading, curv., vel, acc)
              ego_traj_em,  (if available, else ego_traj) planned emergency ego traj (x, y, heading, curv., vel, acc)
              object_array, information about the vehicles in the scene (dict of dicts, each key being the object id and
                            every value holding the following keys ['X', 'Y', 'psi', 'vel', 'length', 'width'])
              time_f)       time-stamps from file (for faster exec.: store this value and hand to function on next iter)
              safety_dyn,   (optional if enabled) safety with respect to other dynamic vehicles [True, False, None]
              safety_stat)  (optional if enabled) safety with respect to a static environment [True, False, None]
    """

    if not (".saa" in file_path or ".scn" in file_path):
        raise ValueError("Unsupported file! Make sure to provide a Scenario-Architect archive ('*.saa') or a scene data"
                         " file ('*.scn').")

    # -- get timestamps ------------------------------------------------------------------------------------------------
    if time_f is None:
        # if archive, extract file
        if ".saa" in file_path:
            with zipfile.ZipFile(file_path) as zipObj:
                tmp_file = next((x for x in zipObj.namelist() if '.scn' in x), None)

                if tmp_file is not None:
                    with zipObj.open(tmp_file) as zf, open(file_path.replace('.saa', '.scn'), 'wb') as f:
                        shutil.copyfileobj(zf, f)
                else:
                    raise ValueError("Could not find *.scn file in the provided Scenario-Architect archive!")

        # load time-stamps from file, if not provided
        time_f = np.genfromtxt(file_path.replace('.saa', '.scn'), delimiter=';', skip_header=2, names=True)['time']

        # if it was in archive, remove extracted file after import
        if ".saa" in file_path:
            os.remove(file_path.replace('.saa', '.scn'))

    if type(t_in) is int:
        idx = t_in
        if idx >= len(time_f):
            raise ValueError("Provided integer time variable is out of the range of the provided file!")
    else:
        # get timestamp before or at position of provided time step
        if time_f[0] <= t_in <= time_f[-1]:
            idx = next((x[0] for x in enumerate(time_f) if x[1] > t_in), time_f.shape[0]) - 1
        else:
            warnings.warn("Provided time value is out of range in provided data file! Returned last time entry.")

            idx = time_f.shape[0] - 1
            t_in = time_f[idx]

    # -- read relevant lines from file ---------------------------------------------------------------------------------
    header = None
    line_prev = None
    line_next = None

    # if archive, extract relevant file
    if ".saa" in file_path:
        zip_obj = zipfile.ZipFile(file_path)
        tmp_file = next((x for x in zip_obj.namelist() if '.scn' in x), None)

        if tmp_file is not None:
            f = zip_obj.open(tmp_file)
            zip_obj.close()
        else:
            raise ValueError("Could not find *.scn file in the provided Scenario-Architect archive!")

    else:
        # directly from file
        f = open(file_path, 'rb')

    i = 0
    while True:
        line = f.readline().decode()
        if i == 2:
            line = line.replace("\n", "").replace("\r", "")
            header = line.split(";")
        elif i == idx + 3:
            line = line.replace("\n", "").replace("\r", "")
            line_prev = line.split(";")
        elif i == idx + 4:
            line = line.replace("\n", "").replace("\r", "")
            line_next = line.split(";")
            break

        i += 1

    f.close()

    # check object_array for proper format
    object_array_prev = json.loads(line_prev[header.index("object_array")])

    if object_array_prev and len(object_array_prev[0][1]) < 6:
        raise ValueError("Provided scenario file does not hold the expected amount of object parameters. Check for "
                         "version compliance (update scenario_testing_tools and scenario-architect to newest version) "
                         "and load + export the relevant scenarios again.")

    # retrieve or interpolate data from file
    if type(t_in) is int or time_f[idx] == t_in or line_next is None:
        # -- extract values at current position ------------------------------------------------------------------------
        time = time_f[idx]
        pos = [float(line_prev[header.index("x")]), float(line_prev[header.index("y")])]
        heading = float(line_prev[header.index("heading")])
        curv = float(line_prev[header.index("curv")])
        vel = float(line_prev[header.index("vel")])
        acc = float(line_prev[header.index("acc")])
        ego_traj = np.array(json.loads(line_prev[header.index("ego_traj")]))
        if 'ego_traj_em' in header:
            ego_traj_em = np.array(json.loads(line_prev[header.index("ego_traj_em")]))
        else:
            ego_traj_em = ego_traj

        object_list = dict()
        for veh in object_array_prev:
            object_list[veh[0]] = {'X': veh[1][0],
                                   'Y': veh[1][1],
                                   'psi': veh[1][2],
                                   'vel': veh[1][3],
                                   'length': veh[1][4],
                                   'width': veh[1][5]}

        safety_dyn = None
        safety_stat = None
        if "safety_dyn" in header and "safety_stat" in header:
            safety_dyn = json.loads(line_prev[header.index("safety_dyn")])
            safety_stat = json.loads(line_prev[header.index("safety_stat")])

    else:
        # -- interpolate between timestamps ----------------------------------------------------------------------------
        time = np.interp(t_in, time_f[idx:idx + 2], time_f[idx:idx + 2])
        pos = [np.interp(t_in, time_f[idx:idx + 2], [float(line_prev[header.index("x")]),
                                                     float(line_next[header.index("x")])]),
               np.interp(t_in, time_f[idx:idx + 2], [float(line_prev[header.index("y")]),
                                                     float(line_next[header.index("y")])])]

        # convert to positive values and back in order to avoid linear interpolation issues
        psi_range = np.array([float(line_prev[header.index("heading")]), float(line_next[header.index("heading")])])
        heading = scenario_testing_tools.interp_heading.interp_heading(heading=psi_range,
                                                                       t_series=time_f[idx:idx + 2],
                                                                       t_in=t_in)

        curv = np.interp(t_in, time_f[idx:idx + 2], [float(line_prev[header.index("curv")]),
                                                     float(line_next[header.index("curv")])])
        vel = np.interp(t_in, time_f[idx:idx + 2], [float(line_prev[header.index("vel")]),
                                                    float(line_next[header.index("vel")])])
        acc = np.interp(t_in, time_f[idx:idx + 2], [float(line_prev[header.index("acc")]),
                                                    float(line_next[header.index("acc")])])

        safety_dyn = None
        safety_stat = None
        if "safety_dyn" in header and "safety_stat" in header:
            sd_prv = json.loads(line_prev[header.index("safety_dyn")])
            sd_nxt = json.loads(line_next[header.index("safety_dyn")])
            safety_dyn = None if sd_prv is None or sd_nxt is None else sd_prv and sd_nxt

            ss_prv = json.loads(line_prev[header.index("safety_stat")])
            ss_nxt = json.loads(line_next[header.index("safety_stat")])
            safety_stat = None if ss_prv is None or ss_nxt is None else ss_prv and ss_nxt

        # get ego-trajectory (interpolate first point for requested t_in)
        ego_traj = np.vstack(
            (interp_1d(x=t_in,
                       xp=time_f[idx:idx + 2],
                       fp_array=np.array([np.array(json.loads(line_prev[header.index("ego_traj")]))[0, :],
                                          np.array(json.loads(line_next[header.index("ego_traj")]))[0, :]]),
                       idx_col_heading=[2]),
             np.array(json.loads(line_next[header.index("ego_traj")])))
        )

        # Check if imported trajectory holds all entries as stated above
        if ego_traj.shape[1] != 6:
            raise ValueError("Could not retrieve all expected trajectory entities (x, y, heading, curv, vel, acc)!")

        # if available, get emergency ego-trajectory (interpolate first point for requested t_in)
        if 'ego_traj_em' in header:
            ego_traj_em = np.vstack(
                (interp_1d(x=t_in,
                           xp=time_f[idx:idx + 2],
                           fp_array=np.array([np.array(json.loads(line_prev[header.index("ego_traj_em")]))[0, :],
                                              np.array(json.loads(line_next[header.index("ego_traj_em")]))[0, :]]),
                           idx_col_heading=[2]),
                 np.array(json.loads(line_next[header.index("ego_traj_em")])))
            )
        else:
            ego_traj_em = ego_traj

        object_array_next = json.loads(line_next[header.index("object_array")])

        object_list = dict()
        for veh_prev, veh_next in zip(object_array_prev, object_array_next):
            interp_obj = interp_1d(x=t_in,
                                   xp=time_f[idx:idx + 2],
                                   fp_array=np.array([veh_prev[1], veh_next[1]]),
                                   idx_col_heading=[2])

            object_list[veh_prev[0]] = {'X': interp_obj[0],
                                        'Y': interp_obj[1],
                                        'psi': interp_obj[2],
                                        'vel': interp_obj[3],
                                        'length': interp_obj[4],
                                        'width': interp_obj[5]}

    if not append_safety:
        return time, pos, heading, curv, vel, acc, ego_traj, ego_traj_em, object_list, time_f
    else:
        return time, pos, heading, curv, vel, acc, ego_traj, ego_traj_em, object_list, time_f, safety_dyn, safety_stat

def interp_heading(heading: np.ndarray,
                   t_series: np.ndarray,
                   t_in: float) -> float:
    """
    Interpolate within a given heading course. (Converts to cos and sin, interpolates and converts back)
    WARNING: Relies on small-angle approximation, i.e. the heading series should be tightly spaced.

    :param heading:             heading course of original data
    :param t_series:            series of time values (matching the heading course)
    :param t_in:                time value for the heading to be generated
    :returns heading_interp:    resulting interpolated heading
    """

    # convert to x, y
    x = np.cos(heading)
    y = np.sin(heading)

    # interpolate in x, y domain
    x1 = np.interp(t_in, t_series, x)
    y1 = np.interp(t_in, t_series, y)

    # convert back to angle
    heading_interp = np.arctan2(y1, x1)

    return heading_interp

def interp_1d(x: float,
              xp: np.ndarray,
              fp_array: np.ndarray,
              idx_col_heading: list = None) -> np.ndarray:
    """
    Extends the functionality of numpy's 'interp' to an array like structure

    :param x:                desired index for new value
    :param xp:               existing series of indexes
    :param fp_array:         array of existing function return values (matching xp)
    :param idx_col_heading:  list of ints holding columns to be treated as headings (avoid jumps when crossing pi/-pi)
    :returns y_array:        array of interpolated values
    """
    y_array = np.zeros(fp_array.shape[1])
    for col in range(fp_array.shape[1]):
        if idx_col_heading is not None and col in idx_col_heading:
            y_array[col] = interp_heading(heading=fp_array[:, col],
                                                                                t_series=xp,
                                                                                t_in=x)
        else:
            y_array[col] = np.interp(x, xp, fp_array[:, col])

    return y_array