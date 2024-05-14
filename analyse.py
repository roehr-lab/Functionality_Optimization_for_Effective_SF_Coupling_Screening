from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 18})

def normal_vector_fit(points):
    # ChatGTP
    points = np.array(points)
    if points.shape[1] != 3:
        raise ValueError("Points must be 3-dimensional")
    # Find the centroid of the points
    centroid = np.mean(points, axis=0)
    # Subtract the centroid from each point
    centered_points = points - centroid
    # Perform SVD on the centered points
    U, S, V = np.linalg.svd(centered_points)
    # The normal to the best-fit plane is the last row of V
    return V[-1]

def get_twist(mol, mol_ref, curve_list, nat_shift = 0):
    nat = len(mol_ref)
    shift = nat * nat_shift
    coords = mol.get_positions()
    coords = np.array(coords)
    mid = len(curve_list)//2
    assert len(curve_list)%2 == 0
    curve_coords = coords[curve_list+shift,:]
    vec_1 = curve_coords[0,:] - curve_coords[-1,:]
    vec_2 = curve_coords[mid-1,:] - curve_coords[mid,:]
    angle_rad = np.arccos(np.clip(np.dot(vec_1, vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2)), -1, 1))
    return np.degrees(angle_rad)

def norm_vector(vec):
    return vec/np.linalg.norm(vec)


def get_bowl(mol, mol_ref, curve_list, nat_shift = 0):
    nat = len(mol_ref)
    shift = nat * nat_shift
    coords = mol.get_positions()
    coords = np.array(coords)
    mid = len(curve_list)//2
    assert len(curve_list)%2 == 0
    curve_coords = coords[curve_list+shift,:]

    vec_short_1 = curve_coords[0,:] - curve_coords[-1,:]
    vec_short_2 = curve_coords[mid-1,:] - curve_coords[mid,:]
    vec_short = np.mean(np.vstack((vec_short_1,vec_short_2)),axis = 0)

    vec_long_1 = np.zeros((mid-1,3))
    for i in range(mid-1):
        vec_long_1[i] = curve_coords[i,:] - curve_coords[i+1,:]

    vec_long_2 = np.zeros((mid-1,3))
    for i in range(mid-1):
        vec_long_2[i] = curve_coords[-1-i,:] - curve_coords[-2-i,:]
    
    vec_long_1 = np.mean(vec_long_1,axis = 0)
    vec_long_2 = np.mean(vec_long_2,axis = 0)

    vec_short = norm_vector(vec_short)
    vec_long_1 = vec_long_1-np.dot(vec_short,vec_long_1)*vec_short
    vec_long_1 = norm_vector(vec_long_1)
    n_1 = np.cross(vec_short,vec_long_1)
    vec_long_2 = vec_long_2-np.dot(vec_short,vec_long_2)*vec_short
    vec_long_2 = norm_vector(vec_long_2)
    n_2 = np.cross(vec_short,vec_long_2)
    x_1 = []
    x_2 = []
    y_1 = []
    y_2 = []
    for i in range(mid):
        x_1.append(np.dot(vec_long_1,curve_coords[i]))
        y_1.append(np.dot(n_1,curve_coords[i]))
        x_2.append(np.dot(vec_long_2,curve_coords[-1-i]))
        y_2.append(np.dot(n_2,curve_coords[-1-i]))

    x_1 = np.array(x_1)
    y_1 = np.array(y_1)

    x_2 = np.array(x_2)
    y_2 = np.array(y_2)

    fit_1 = np.polyfit(x_1,y_1,2)
    fit_2 = np.polyfit(x_2,y_2,2)

    return (fit_1[0], fit_2[0])

def get_basis_vector(curve_coords):
    mid = len(curve_coords)//2
    assert len(curve_coords)%2 == 0

    vec_short_1 = curve_coords[0,:] - curve_coords[-1,:]
    vec_short_2 = curve_coords[mid-1,:] - curve_coords[mid,:]
    vec_short = np.mean(np.vstack((vec_short_1,vec_short_2)),axis = 0)

    vec_long_1 = np.zeros((mid-1,3))
    for i in range(mid-1):
        vec_long_1[i] = curve_coords[i,:] - curve_coords[i+1,:]

    vec_long_2 = np.zeros((mid-1,3))
    for i in range(mid-1):
        vec_long_2[i] = curve_coords[-1-i,:] - curve_coords[-2-i,:]
    
    vec_long_1 = np.mean(vec_long_1,axis = 0)
    vec_long_2 = np.mean(vec_long_2,axis = 0)
    vec_long = np.mean(np.vstack((vec_long_1,vec_long_1)),axis = 0)

    vec_long = norm_vector(vec_long)
    vec_short = vec_short-np.dot(vec_long,vec_short)*vec_long
    vec_short = norm_vector(vec_short)
    vec_normal = np.cross(vec_long, vec_short)
    vec_normal = norm_vector(vec_normal)

    return np.vstack((vec_long,vec_short,vec_normal))

def get_center_of_mass(coords, species):
    return np.array([np.sum(coords[:,0]*species)/np.sum(species),np.sum(coords[:,1]*species)/np.sum(species),np.sum(coords[:,2]*species)/np.sum(species)])

def to_angles_0_to_90(degrees):
    # Apply modulo 360 to ensure angles are in the range [0, 360)
    degrees = np.array(degrees)
    degrees = np.mod(degrees, 360)

    # Convert angles larger than 90 to their complementary angle
    complementary_angles = np.where(degrees > 180, degrees-180, degrees)
    complementary_angles = np.where(complementary_angles > 90, 180 - complementary_angles, complementary_angles)
    return complementary_angles

def to_angles_0_180(degrees):
    degrees = np.array(degrees)
    degrees = np.mod(degrees, 360)

    # Convert angles larger than 90 to their complementary angle
    complementary_angles = np.where(degrees > 180, degrees-180, degrees)
    return complementary_angles

def get_trans_rot_dimer(mol, mol_ref, curve_list, dimer_0 = 0, dimer_1 = 1, raw_angles = False):
    nat = len(mol_ref)
    shift_0 = nat * dimer_0
    shift_1 = nat * dimer_1
    coords = mol.get_positions()
    coords = np.array(coords)
    species = mol.get_atomic_numbers()
    species = np.array(species)
    curve_coords_0 = coords[curve_list+shift_0,:]
    curve_coords_1 = coords[curve_list+shift_1,:]
    basis_0 = get_basis_vector(curve_coords_0)
    basis_1 = get_basis_vector(curve_coords_1)
    c_mass_0 = get_center_of_mass(coords[shift_0:shift_0+nat],species[shift_0:shift_0+nat])
    c_mass_1 = get_center_of_mass(coords[shift_1:shift_1+nat],species[shift_1:shift_1+nat])

    if np.dot(basis_0[2],basis_1[2]) < 0:
        basis_1[1:] = -basis_1[1:]

    c_mass_diff = c_mass_0 - c_mass_1

    d_x = np.dot(basis_0[0],c_mass_diff)
    d_y = np.dot(basis_0[1],c_mass_diff)
    d_z = np.dot(basis_0[2],c_mass_diff)


    if not np.allclose(np.dot(basis_0[1],basis_1[1]),0):
        dx_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[1],basis_1[1])/np.linalg.norm(np.array([np.dot(basis_0[1],basis_1[1]),np.dot(basis_0[2],basis_1[1])])),-1,1)))
    elif not np.allclose(np.dot(basis_0[2],basis_1[2]),0.):
        dx_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[2],basis_1[2])/np.linalg.norm(np.array([np.dot(basis_0[1],basis_1[2]),np.dot(basis_0[2],basis_1[2])])),-1,1)))
    else:
        dx_rot = 0
    

    if not np.allclose(np.dot(basis_0[2],basis_1[2]),0.):
        dy_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[2],basis_1[2])/np.linalg.norm(np.array([np.dot(basis_0[2],basis_1[2]),np.dot(basis_0[0],basis_1[2])])),-1,1)))
    elif not np.allclose(np.dot(basis_0[0],basis_1[0]),0.):
        dy_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[0],basis_1[0])/np.linalg.norm(np.array([np.dot(basis_0[2],basis_1[0]),np.dot(basis_0[0],basis_1[0])])),-1,1)))
    else:
        dy_rot = 0
    

    if not np.allclose(np.dot(basis_0[0],basis_1[0]),0.):
        dz_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[0],basis_1[0])/np.linalg.norm(np.array([np.dot(basis_0[0],basis_1[0]),np.dot(basis_0[1],basis_1[0])])),-1,1)))
    elif not np.allclose(np.dot(basis_0[1],basis_1[1]),0.):
        dz_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[1],basis_1[1])/np.linalg.norm(np.array([np.dot(basis_0[0],basis_1[1]),np.dot(basis_0[1],basis_1[1])])),-1,1)))
    else:
        dz_rot = 0
    

    if not raw_angles:
        dx_rot = float(to_angles_0_180(dx_rot))
        dy_rot = float(to_angles_0_180(dy_rot))
        dz_rot = float(to_angles_0_180(dz_rot))
    
    return (d_x,d_y,d_z,dx_rot,dy_rot,dz_rot)

def analyse_PBI(path_monomer_structure, path_dimers, path_scan_data, mode = "best_score", table_head = "latex/table_head_PBI.txt", outputfile = "latex/PBI_table.tex"):
    curve_list = np.array([15,8,5,4,23,27,26,22,2,3,9,10])-1

    data = np.load(path_scan_data)['data']
    traj_len = np.zeros((len(data)))
    for i, d_i in enumerate(data):
        traj_len[i] = np.size(d_i[:,0]) - np.count_nonzero(np.isnan(d_i[:,0]))
    score = data[:,:,2]
    sfr = data[:,:,1]
    d_e = data[:,:,0]
    score[np.isnan(score)] = 1000
    sfr[np.isnan(sfr)] = 1e-31
    d_e[np.isnan(d_e)] = 60
    data_best_score = np.min(score,axis = 1)
    data_best_sfr = np.max(sfr,axis = 1)
    data_best_d_e = np.min(d_e,axis = 1)
    data_best_score_arg = np.argmin(score,axis = 1)
    data_best_sfr_arg = np.argmax(sfr,axis = 1)
    data_best_d_e_arg = np.argmin(d_e,axis = 1)

    if mode == "best_score":
        sfr_best_score = np.zeros_like(traj_len)
        for i in range(len(traj_len)):
                sfr_best_score[i] = sfr[i,data_best_score_arg[i]]
        sort = np.argsort(-sfr_best_score)
        sort_array = sfr_best_score
    elif mode == "best_sfr":
        sort = np.argsort(-data_best_sfr)
        sort_array = data_best_sfr
    else:
        raise NotImplementedError("This mode is not implemented!")
    
    mol_ref = read(path_monomer_structure, format='xyz',index=":")[0]
    mol_all_dimers = read(path_dimers, format='xyz',index=":")

    with open(table_head,"r") as head:
        output = head.readlines()
    
    linear = np.arange(len(mol_all_dimers))
    for idx_dimer, dimer_mol in enumerate(mol_all_dimers):
        # get_RMSE_binding_length_character(dimer_mol, mol_ref, nat_shift=0))
        steps = traj_len[sort[idx_dimer]]-2
        if mode == "best_score":
            d_E = d_e[sort[idx_dimer],data_best_score_arg[sort[idx_dimer]]]
        elif mode == "best_sfr":
            d_E = d_e[sort[idx_dimer],data_best_sfr_arg[sort[idx_dimer]]]
        rate = sort_array[sort[idx_dimer]]
        coords = dimer_mol.get_positions()
        coords = np.array(coords)
        curve_coords_0 = coords[curve_list,:]
        curve_coords_1 = coords[curve_list+len(mol_ref),:]
        basis_0 = get_basis_vector(curve_coords_0)
        basis_1 = get_basis_vector(curve_coords_1)
        twist_a = get_twist(dimer_mol, mol_ref, curve_list,0)
        twist_b = get_twist(dimer_mol, mol_ref, curve_list,1)
        (bowl_a1, bowl_a2) = get_bowl(dimer_mol, mol_ref, curve_list,0)
        bowl_a = np.mean(np.array([bowl_a1, bowl_a2]))
        (bowl_b1, bowl_b2) = get_bowl(dimer_mol, mol_ref, curve_list,1)
        bowl_b = np.mean(np.array([bowl_b1, bowl_b2]))*np.sign(np.dot(basis_0[2],basis_1[2]))
        (d_x,d_y,d_z,dx_rot,dy_rot,dz_rot) = get_trans_rot_dimer(dimer_mol, mol_ref, curve_list)
        outputline = f"\\num{{{idx_dimer+1}}} & \\num{{{linear[sort[idx_dimer]]}}} & \\num{{{d_x:.2f}}} & \\num{{{d_y:.2f}}} & \\num{{{d_z:.2f}}} & \\num{{{dx_rot:.1f}}} & \\num{{{dy_rot:.1f}}} & \\num{{{dz_rot:.1f}}} & \\num{{{bowl_a*2e3:.1f}}} & \\num{{{bowl_b*2e3:.1f}}} & \\num{{{twist_a:.1f}}} & \\num{{{twist_b:.1f}}} & \\num{{{rate:.2e}}} & \\num{{{d_E:.2f}}} & \\num{{{steps:.0f}}}\\\\\n"
        output.append(outputline)
        output.append("\\hline\n")

    output.append("\end{longtable}\n")
    output.append("\end{landscape}\n")

    with open(outputfile, "w") as out:
        out.writelines(output)

def statistic_PBI(path_monomer_structure, path_dimers, path_scan_data, mode = "best_score", write_PCA = False, n_clusters = 4, n_components = 3, ignore_dz = True, ignore_rot_dx = True, ignore_rot_dy = True, use_absolutes = True, out_file="analysis/analysis_new.txt"):
    info_out = []

    curve_list = np.array([15,8,5,4,23,27,26,22,2,3,9,10])-1

    data = np.load(path_scan_data)['data']
    traj_len = np.zeros((len(data)))
    for i, d_i in enumerate(data):
        traj_len[i] = np.size(d_i[:,0]) - np.count_nonzero(np.isnan(d_i[:,0]))
    score = data[:,:,2]
    sfr = data[:,:,1]
    d_e = data[:,:,0]
    score[np.isnan(score)] = 1000
    sfr[np.isnan(sfr)] = 1e-31
    d_e[np.isnan(d_e)] = 60
    data_best_score = np.min(score,axis = 1)
    data_best_sfr = np.max(sfr,axis = 1)
    data_best_d_e = np.min(d_e,axis = 1)
    data_best_score_arg = np.argmin(score,axis = 1)
    data_best_sfr_arg = np.argmax(sfr,axis = 1)
    data_best_d_e_arg = np.argmin(d_e,axis = 1)

    if mode == "best_score":
        sfr_best_score = np.zeros_like(traj_len)
        for i in range(len(traj_len)):
                sfr_best_score[i] = sfr[i,data_best_score_arg[i]]
        sort = np.argsort(-sfr_best_score)
        sort_array = sfr_best_score
    elif mode == "best_sfr":
        sort = np.argsort(-data_best_sfr)
        sort_array = data_best_sfr
    else:
        raise NotImplementedError("This mode is not implemented!")
    
    mol_ref = read(path_monomer_structure, format='xyz',index=":")[0]
    mol_all_dimers = read(path_dimers, format='xyz',index=":")

    steps_list = []
    d_E_list = []
    rate_list = []
    twist_a_list = []
    twist_b_list = []
    bowl_a_list = []
    bowl_b_list = []
    d_x_list = []
    d_y_list = []
    d_z_list = []
    dx_rot_list = []
    dy_rot_list = []
    dz_rot_list = []
    score_list = []

    border = 239
    info_out.append(f"Used structures: 1-{border}\n")
    info_out.append(f"Used structures: {(border)/len(mol_all_dimers)*100:.1f}%\n")


    for idx_dimer, dimer_mol in enumerate(mol_all_dimers[:border]):
        # get_RMSE_binding_length_character(dimer_mol, mol_ref, nat_shift=0))
        steps = traj_len[sort[idx_dimer]]-2
        if mode == "best_score":
            d_E = d_e[sort[idx_dimer],data_best_score_arg[sort[idx_dimer]]]
            sc = score[sort[idx_dimer],data_best_score_arg[sort[idx_dimer]]]
        elif mode == "best_sfr":
            d_E = d_e[sort[idx_dimer],data_best_sfr_arg[sort[idx_dimer]]]
            sc = score[sort[idx_dimer],data_best_sfr_arg[sort[idx_dimer]]]
        rate = sort_array[sort[idx_dimer]]
        coords = dimer_mol.get_positions()
        coords = np.array(coords)
        curve_coords_0 = coords[curve_list,:]
        curve_coords_1 = coords[curve_list+len(mol_ref),:]
        basis_0 = get_basis_vector(curve_coords_0)
        basis_1 = get_basis_vector(curve_coords_1)
        twist_a = get_twist(dimer_mol, mol_ref, curve_list,0)
        twist_b = get_twist(dimer_mol, mol_ref, curve_list,1)
        (bowl_a1, bowl_a2) = get_bowl(dimer_mol, mol_ref, curve_list,0)
        bowl_a = np.mean(np.array([bowl_a1, bowl_a2]))
        (bowl_b1, bowl_b2) = get_bowl(dimer_mol, mol_ref, curve_list,1)
        bowl_b = np.mean(np.array([bowl_b1, bowl_b2]))*np.sign(np.dot(basis_0[2],basis_1[2]))
        (d_x,d_y,d_z,dx_rot,dy_rot,dz_rot) = get_trans_rot_dimer(dimer_mol, mol_ref, curve_list, raw_angles=True)
        steps_list.append(steps)
        d_E_list.append(d_E)
        rate_list.append(rate)
        twist_a_list.append(twist_a)
        twist_b_list.append(twist_b)
        bowl_a_list.append(bowl_a)
        bowl_b_list.append(bowl_b)
        d_x_list.append(d_x)
        d_y_list.append(d_y)
        d_z_list.append(d_z)
        dx_rot_list.append(dx_rot)
        dy_rot_list.append(dy_rot)
        dz_rot_list.append(dz_rot)
        score_list.append(sc)


    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    n_attributes = 10
    X = np.zeros((len(twist_a_list),n_attributes))
    X[:,8] = np.array(twist_a_list)
    X[:,9] = np.array(twist_b_list)
    X[:,6] = np.array(bowl_a_list)*2e3
    X[:,7] = np.array(bowl_b_list)*2e3
    X[:,0] = np.array(d_x_list)
    X[:,1] = np.array(d_y_list)
    X[:,2] = np.array(d_z_list)
    X[:,3] = to_angles_0_180(np.array(dx_rot_list))
    X[:,4] = to_angles_0_180(np.array(dy_rot_list))
    X[:,5] = to_angles_0_180(np.array(dz_rot_list))

    energy = np.array(d_E_list)
    info_out.append(f"________________________________\n")
    for name, array in [("SFR", sfr), ("delta_E", energy), ("dx",X[:,0]),("dy",X[:,1]),("dz",X[:,2]), ("abs(dx)",np.abs(X[:,0])), ("abs(dy)",np.abs(X[:,1])), ("abs(dz)",np.abs(X[:,2])), ("rot x",X[:,3]), ("rot y",X[:,4]), ("rot z",X[:,5]), ("abs(rot x)",to_angles_0_to_90(X[:,3])), ("abs(rot y)",to_angles_0_to_90(X[:,4])), ("abs(rot z)",to_angles_0_to_90(X[:,5])), ("concav a",X[:,6]), ("concav b",X[:,7]), ("twist a",X[:,8]), ("twist b",X[:,9]) ]:
        info_out.append(f"Mean {name}: {np.mean(array)}\n")
        info_out.append(f"Std {name}: {np.std(array)}\n")
        info_out.append(f"Max {name}: {np.max(array)}\n")
        info_out.append(f"Min {name}: {np.min(array)}\n")

    info_out.append(f"________________________________\n")
    if ignore_dz:
        info_out.append(f"Set all dz to 3\n")
        X[:,2] = 3
    if ignore_rot_dx:
        info_out.append(f"Set all dx_rot to 0\n")
        X[:,3] = 0
    if ignore_rot_dy:
        info_out.append(f"Set all dy_rot to 0\n")
        X[:,4] = 0
    if use_absolutes:
        info_out.append(f"Transform all dz_rot to -90 to 90\n")
        X[:,5] = to_angles_0_to_90(X[:,5]) * np.sign(X[:,0]) * np.sign(X[:,1])
        info_out.append(f"Transform all dx to abs(dx)\n")
        X[:,0] = np.abs(X[:,0])
        info_out.append(f"Transform all dy to abs(dy)\n")
        X[:,1] = np.abs(X[:,1])

    info_out.append(f"________________________________\n")
    info_out.append(f"Amount Clusters: {n_clusters}\n")
    info_out.append(f"PCA Components: {n_components}\n")

    original_X = np.copy(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    transformed_data = pca.transform(X)
    explained_variance_ratios = np.array(pca.explained_variance_ratio_)
    # Get the principal components
    components = np.array(pca.components_)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(transformed_data)
    labels = np.array(kmeans.labels_)
    centroids = np.array(kmeans.cluster_centers_)

    if write_PCA:
        np.savez(out_file, components = components, labels = labels, centroids = centroids, n_attributes = np.array(n_attributes), original_X = original_X, n_clusters = np.array(n_clusters), X = X, rate_list = np.array(rate_list), transformed_data  =transformed_data, explained_variance_ratios = explained_variance_ratios, energy = energy, steps = np.array(steps_list), n_components = np.array(n_components))

    with open("analysis/analysis_1.txt",  "w") as file:
        file.writelines(info_out)

def latex_for_PCA_analysis(file_name = "analysis/analysis_new.npz", n_clusters = 4, n_components = 3, ignore_dz = True, ignore_rot_dx = True, ignore_rot_dy = True, use_absolutes = True, ):
    attributes = ['translation x', 'translation y', 'translation z', 'rotation x', 'rotation y', 'rotation z', 'concavity a', 'concavity b', 'twist a', 'twist b']
    latex_out = []


    data = np.load(file_name)
    components = data["components"]
    labels = data["labels"]
    centroids = data["centroids"]
    n_attributes = int(data["n_attributes"])
    original_X = data["original_X"]
    n_clusters = int(data["n_clusters"])
    continuous_values = data["rate_list"]
    transformed_data = data["transformed_data"]
    explained_variance_ratios = data["explained_variance_ratios"]
    sfr = continuous_values
    n_components = int(data["n_components"])

    mean_centers = np.zeros((n_clusters,n_components))
    std_centers = np.zeros((n_clusters,n_components))
    for i in range(n_clusters):
        mean_centers[i] = np.mean(transformed_data[i == labels] ,axis = 0)
        std_centers[i] = np.std(transformed_data[i == labels], axis = 0)

    character_of_clusters = np.einsum("ij,jl -> il", mean_centers, components)

    renormed_character_of_cluster = np.copy(character_of_clusters)
    renormed_character_of_cluster_std = np.einsum("ij,jl -> il", std_centers, components)
    for i in range(n_attributes):
        renormed_character_of_cluster[:,i] = renormed_character_of_cluster[:,i]*np.std(original_X[:,i])+np.mean(original_X[:,i])
        renormed_character_of_cluster_std[:,i] = np.abs(renormed_character_of_cluster_std[:,i]*np.std(original_X[:,i]))

    pca_dim_str = ""

    # Print explained variance ratios
    for i, ratio in enumerate(explained_variance_ratios):
        pca_dim_str+=f"Dimension {i+1}: {ratio*100:.0f}\\% of the variance. "


    with open("latex/table_head_PBI_PCA.txt","r") as head:
        latex_out = head.readlines()

    latex_out[2] = latex_out[2].replace("$dim$",str(n_components))
    latex_out[2] = latex_out[2].replace("$clusters$",str(n_clusters))
    latex_out[2] = latex_out[2].replace("$ztrans$",str(ignore_dz))
    latex_out[2] = latex_out[2].replace("$xrot$",str(ignore_rot_dx))
    latex_out[2] = latex_out[2].replace("$yrot$",str(ignore_rot_dy))
    latex_out[2] = latex_out[2].replace("$sym$",str(use_absolutes))
    latex_out[2] = latex_out[2].replace("$PCADIM$",pca_dim_str)


    for i in range(n_clusters):
        rate = np.mean(sfr[labels == i])
        d_x, d_y, d_z, dx_rot, dy_rot, dz_rot, bowl_a, bowl_b, twist_a, twist_b = renormed_character_of_cluster[i,:]

        
        outputline = f"\\num{{{i+1}}} & \\num{{{d_x:.2f}}} & \\num{{{d_y:.2f}}} & \\num{{{d_z:.2f}}} & \\num{{{dx_rot:.1f}}} & \\num{{{dy_rot:.1f}}} & \\num{{{dz_rot:.1f}}} & \\num{{{bowl_a:.1f}}} & \\num{{{bowl_b:.1f}}} & \\num{{{twist_a:.1f}}} & \\num{{{twist_b:.1f}}} & \\num{{{rate:.2e}}} \\\\\n"
        latex_out.append(outputline)
        latex_out.append("\\hline\n")

    latex_out.append("\end{longtable}\n")
    latex_out.append("\end{landscape}\n")

    return latex_out


def plot_PBI_analysis():
    colors = ['g', 'b', 'y', 'c', 'm', 'k', 'skyblue', 'goldenrod', 'crimson', 'lightgreen']
    attributes = ['translation x', 'translation y', 'translation z', 'rotation x', 'rotation y', 'rotation z', 'concavity a', 'concavity b', 'twist a', 'twist b']
    info_out = []


    data = np.load("analysis/analysis.npz")
    components = data["components"]
    labels = data["labels"]
    centroids = data["centroids"]
    n_attributes = int(data["n_attributes"])
    original_X = data["original_X"]
    n_clusters = int(data["n_clusters"])
    continuous_values = data["rate_list"]
    transformed_data = data["transformed_data"]
    explained_variance_ratios = data["explained_variance_ratios"]
    sfr = continuous_values
    n_components = int(data["n_components"])

    mean_centers = np.zeros((n_clusters,n_components))
    std_centers = np.zeros((n_clusters,n_components))
    for i in range(n_clusters):
        mean_centers[i] = np.mean(transformed_data[i == labels] ,axis = 0)
        std_centers[i] = np.std(transformed_data[i == labels], axis = 0)

    character_of_clusters = np.einsum("ij,jl -> il", mean_centers, components)

    renormed_character_of_cluster = np.copy(character_of_clusters)
    renormed_character_of_cluster_std = np.einsum("ij,jl -> il", std_centers, components)
    for i in range(n_attributes):
        renormed_character_of_cluster[:,i] = renormed_character_of_cluster[:,i]*np.std(original_X[:,i])+np.mean(original_X[:,i])
        renormed_character_of_cluster_std[:,i] = np.abs(renormed_character_of_cluster_std[:,i]*np.std(original_X[:,i]))
    info_out.append(f"________________________________\n")
    for i in range(n_clusters):
        info_out.append(f"Group  {i} Color: {colors[i]}\n")
        info_out.append(f"Mean SFR: {np.mean(sfr[labels == i])}\n")
        for j, atr in enumerate(attributes):
            info_out.append(f"{atr}:  {renormed_character_of_cluster[i,j]}\n")
            info_out.append(f"std({atr}):  {renormed_character_of_cluster_std[i,j]}\n")
        info_out.append(f"________________________________\n")
        make_dimers_from_trans_rot("/Users/johannes/Documents/Uni/Masterarbeit/Projects/ML_for_SF/twisty/pbi_flat.xyz",renormed_character_of_cluster[i,:],f"analysis/struct_{i:04d}")



    # Print explained variance ratios
    for i, ratio in enumerate(explained_variance_ratios):
        info_out.append(f"Dimension {i+1}: {ratio:.2%} of the variance\n")

    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=np.min(continuous_values), vmax=np.max(continuous_values))
    normalized_continuous_values = norm(continuous_values)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(32, 8), sharex=True)

    vmax = np.max(np.abs(character_of_clusters))
    
    
    for i, ax in enumerate(axes):
        ax.bar(attributes, character_of_clusters[i], color=colors[i])
        ax.set_ylim([-vmax, vmax])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
        # Hide the y-axis
        ax.yaxis.set_visible(False)

    fig.savefig("analysis/contributions.png", dpi = 300)
    from matplotlib.cm import ScalarMappable

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    for i in range(n_clusters):
        sel = labels == i
        ax.scatter(transformed_data[sel, 0], transformed_data[sel, 1], transformed_data[sel, 2],s=300,c=colors[i], alpha = 0.4, label = f"cluster {i+1}")
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2],s=20,c=normalized_continuous_values, cmap='Reds')
    ax.view_init(25,45)
    ax.set_xlabel('PCA dimension 1', labelpad=10)
    ax.set_ylabel('PCA dimension 2', labelpad=10)
    ax.set_zlabel('PCA dimension 3', labelpad=10)
    ax.legend(loc=2)
    ax.set_zlim([-2.5,3.5])
    sm = ScalarMappable(cmap='Reds')
    sm.set_array(continuous_values)
    plt.colorbar(sm, ax=ax, label='SF rate')
    fig.savefig("analysis/3D_cluster.png", dpi=300)
    

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    for i in range(n_clusters):
        sel = labels == i
        ax.scatter(transformed_data[sel, 0], transformed_data[sel, 1],s=200,c=colors[i], label = f"cluster {i+1}")
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], c=normalized_continuous_values, cmap='Reds')
    ax.set_xlabel('PCA dimension 1')
    ax.set_ylabel('PCA dimension 2')
    ax.legend()
    sm = ScalarMappable(cmap='Reds')
    sm.set_array(continuous_values)
    plt.colorbar(sm, ax=ax, label='SF rate')
    fig.savefig("analysis/2D_cluster.png", dpi = 300)

    with open("analysis/analysis_2.txt",  "w") as file:
        file.writelines(info_out)

def make_dimers_from_trans_rot(input_structure, attributes, output_name, comment = ""):

    from ase.io.extxyz import write_extxyz
    from ase.atoms import Atoms

    monomer = read(input_structure)

    monomer_coords = np.array(monomer._get_positions())
    monomer_species = monomer.get_chemical_symbols()
    monomer_nat = len(monomer_species)

    dimer_coords = np.zeros((monomer_nat*2,3))
    dimer_species = []

    dx = attributes[0]
    dy = attributes[1]
    dz = attributes[2]
    dx_rot = attributes[3]*np.pi/180
    dy_rot = attributes[4]*np.pi/180
    dz_rot = attributes[5]*np.pi/180

    for i in range(monomer_nat):
        dimer_species.append(monomer_species[i])
        dimer_coords[i,:] = monomer_coords[i,:]
        dimer_coords[i,2] += np.tan(np.radians(attributes[8])/10) * monomer_coords[i,0]*monomer_coords[i,1]
        dimer_coords[i,2] += attributes[6] /2e3 * monomer_coords[i,0]*monomer_coords[i,0]
    for i in range(monomer_nat):
        dimer_species.append(monomer_species[i])
        x,y,z = monomer_coords[i,:]
        z += x*y*np.tan(np.radians(attributes[9]/10))
        z += attributes[7] /2e3 * x*x
        x_n = x*(np.cos(dy_rot)*np.cos(dz_rot)) + y*(-np.cos(dx_rot)*np.sin(dz_rot) + np.sin(dx_rot)*np.sin(dy_rot)*np.cos(dz_rot)) + z*(np.sin(dx_rot)*np.sin(dz_rot)+np.cos(dx_rot)*np.sin(dy_rot)*np.cos(dz_rot))
        y_n = x*(np.cos(dy_rot)*np.sin(dz_rot)) + y*(np.cos(dx_rot)*np.cos(dz_rot) + np.sin(dx_rot)*np.sin(dy_rot)*np.sin(dz_rot)) + z*(-np.sin(dx_rot)*np.cos(dz_rot) + np.cos(dx_rot)*np.sin(dy_rot)*np.sin(dz_rot))
        z_n = x*(-np.sin(dy_rot)) + y*(np.sin(dx_rot)*np.cos(dy_rot)) + z*(np.cos(dx_rot)*np.cos(dy_rot))
        dimer_coords[i+monomer_nat,:] = np.array([x_n+dx,y_n+dy,z_n+dz])

    dimer = Atoms(dimer_species, dimer_coords)
    write_extxyz(f"{output_name}.xyz", dimer, comment = comment)

def generate_caption():
    data = np.load("analysis/analysis.npz")
    components = data["components"]
    labels = data["labels"]
    n_attributes = int(data["n_attributes"])
    original_X = data["original_X"]
    n_clusters = int(data["n_clusters"])
    transformed_data = data["transformed_data"]
    n_components = int(data["n_components"])

    mean_centers = np.zeros((n_clusters,n_components))
    var_centers = np.zeros((n_clusters,n_components))
    for i in range(n_clusters):
        mean_centers[i] = np.mean(transformed_data[i == labels] ,axis = 0)
        var_centers[i] = np.var(transformed_data[i == labels], axis = 0)


    character_of_clusters = np.einsum("ij,jl -> il", mean_centers, components)

    renormed_character_of_cluster = np.copy(character_of_clusters)
    renormed_character_of_cluster_var = np.einsum("ij,jl -> il", var_centers, components)
    for i in range(n_attributes):
        renormed_character_of_cluster[:,i] = renormed_character_of_cluster[:,i]*np.std(original_X[:,i])+np.mean(original_X[:,i])
        renormed_character_of_cluster_var[:,i] = np.abs(renormed_character_of_cluster_var[:,i]*np.std(original_X[:,i]))

    for i in range(n_clusters):
        print(i+1)
        print(f"The dimer features a translation in x of \\SI{{{renormed_character_of_cluster[i,0]:.2f}}}{{\\angstrom}} and in y of \\SI{{{renormed_character_of_cluster[i,1]:.2f}}}{{\\angstrom}}, a z-rotation of \\SI{{{renormed_character_of_cluster[i,5]:.1f}}}{{\\degree}}, concavities of \\SI{{{renormed_character_of_cluster[i,6]:.1f}e-3}}{{\\per\\angstrom\\squared}} and \\SI{{{renormed_character_of_cluster[i,7]:.1f}e-3}}{{\\per\\angstrom\\squared}} and twisting angles of \\SI{{{renormed_character_of_cluster[i,8]:.1f}}}{{\\degree}} and \\SI{{{renormed_character_of_cluster[i,9]:.1f}}}{{\\degree}} on monomer A and B, respectively.")

def assemble_info_PBI(path_scan_data, mode = "best_score"):
    data = np.load(path_scan_data)['data']
    traj_len = np.zeros((len(data)))
    for i, d_i in enumerate(data):
        traj_len[i] = np.size(d_i[:,0]) - np.count_nonzero(np.isnan(d_i[:,0]))
    score = data[:,:,2]
    sfr = data[:,:,1]
    d_e = data[:,:,0]
    score[np.isnan(score)] = 1000
    sfr[np.isnan(sfr)] = 1e-31
    d_e[np.isnan(d_e)] = 60
    data_best_score = np.min(score,axis = 1)
    data_best_sfr = np.max(sfr,axis = 1)
    data_best_d_e = np.min(d_e,axis = 1)
    data_best_score_arg = np.argmin(score,axis = 1)
    data_best_sfr_arg = np.argmax(sfr,axis = 1)
    data_best_d_e_arg = np.argmin(d_e,axis = 1)

    if mode == "best_score":
        sfr_best_score = np.zeros_like(traj_len)
        for i in range(len(traj_len)):
                sfr_best_score[i] = sfr[i,data_best_score_arg[i]]
        sort = np.argsort(-sfr_best_score)
        sort_array = sfr_best_score
    elif mode == "best_sfr":
        sort = np.argsort(-data_best_sfr)
        sort_array = data_best_sfr
    else:
        raise NotImplementedError("This mode is not implemented!")
    
    print(f"finished trajectories: {np.count_nonzero(traj_len<302)}/{len(traj_len)}")
    print(f"mean steps of finished trajectories: {np.mean(traj_len[traj_len<302])}")

    bins = np.linspace(-27,-6,22)
    print(bins)

    log_10_sfr = np.log(sort_array)/np.log(10)
    log_10_sfr_init = np.log(sfr[:,0])/np.log(10)


    print(f"Begin mean sfr {np.mean(sfr[:,0])}")
    print(f"End mean sfr {np.mean(sort_array)}")


    fig = plt.figure(figsize=(8,5))
    
    plt.rc('axes', axisbelow=True)#
    plt.grid(axis='y')
    total = len(log_10_sfr)
    bin_counts, _, bin_list = plt.hist(log_10_sfr, bins=bins, color="red", alpha=0.7)
    bin_list[-1].set_fc("orange")
    for count, bin_edge in zip(bin_counts, bins[:-1]):
        percentage = count / total * 100
        #plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, f'{count:.0f}', ha='center', va='bottom', color = "red", fontsize = 14)
    bin_counts, _, bin_list = plt.hist(log_10_sfr_init, bins=bins, color="blue", alpha=0.7)
    for count, bin_edge in zip(bin_counts, bins[:-1]):
        percentage = count / total * 100
        #plt.text(bin_edge + (bins[1] - bins[0]) / 2, -15, f'{count:.0f}', ha='center', va='bottom', color = "blue", fontsize = 14)
        #sel = np.logical_and(log_10_sfr > bin_edge, log_10_sfr < bin_edge+1)
        #amount_failed = np.count_nonzero(traj_len[sel] > 301)
        #plt.text(bin_edge + (bins[1] - bins[0]) / 2, 270, f'{amount_failed}', ha='center', va='bottom', color = "blue")
    # Add labels and title
    plt.xlabel(r'$log_{10}(|T_{RP}|^2)$')
    plt.ylabel('structures')
    plt.ylim([0,250])
    plt.xlim([-27,-6])
    
    plt.tight_layout()
    plt.savefig("analysis/histogram.png",dpi=300)

def get_trans_rot_dimer_from_lists(mol, curve_list_1, curve_list_2, raw_angles = False):
    coords = mol.get_positions()
    coords = np.array(coords)
    species = mol.get_atomic_numbers()
    species = np.array(species)
    curve_coords_0 = coords[curve_list_1,:]
    curve_coords_1 = coords[curve_list_2,:]
    basis_0 = get_basis_vector(curve_coords_0)
    basis_1 = get_basis_vector(curve_coords_1)
    c_mass_0 = get_center_of_mass(coords[curve_list_1],species[curve_list_1])
    c_mass_1 = get_center_of_mass(coords[curve_list_2],species[curve_list_2])

    if np.dot(basis_0[2],basis_1[2]) < 0:
        basis_1[1:] = -basis_1[1:]

    c_mass_diff = c_mass_0 - c_mass_1

    d_x = np.dot(basis_0[0],c_mass_diff)
    d_y = np.dot(basis_0[1],c_mass_diff)
    d_z = np.dot(basis_0[2],c_mass_diff)


    if not np.allclose(np.dot(basis_0[1],basis_1[1]),0):
        dx_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[1],basis_1[1])/np.linalg.norm(np.array([np.dot(basis_0[1],basis_1[1]),np.dot(basis_0[2],basis_1[1])])),-1,1)))
    elif not np.allclose(np.dot(basis_0[2],basis_1[2]),0.):
        dx_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[2],basis_1[2])/np.linalg.norm(np.array([np.dot(basis_0[1],basis_1[2]),np.dot(basis_0[2],basis_1[2])])),-1,1)))
    else:
        dx_rot = 0
    

    if not np.allclose(np.dot(basis_0[2],basis_1[2]),0.):
        dy_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[2],basis_1[2])/np.linalg.norm(np.array([np.dot(basis_0[2],basis_1[2]),np.dot(basis_0[0],basis_1[2])])),-1,1)))
    elif not np.allclose(np.dot(basis_0[0],basis_1[0]),0.):
        dy_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[0],basis_1[0])/np.linalg.norm(np.array([np.dot(basis_0[2],basis_1[0]),np.dot(basis_0[0],basis_1[0])])),-1,1)))
    else:
        dy_rot = 0
    

    if not np.allclose(np.dot(basis_0[0],basis_1[0]),0.):
        dz_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[0],basis_1[0])/np.linalg.norm(np.array([np.dot(basis_0[0],basis_1[0]),np.dot(basis_0[1],basis_1[0])])),-1,1)))
    elif not np.allclose(np.dot(basis_0[1],basis_1[1]),0.):
        dz_rot = np.degrees(np.arccos(np.clip(np.dot(basis_0[1],basis_1[1])/np.linalg.norm(np.array([np.dot(basis_0[0],basis_1[1]),np.dot(basis_0[1],basis_1[1])])),-1,1)))
    else:
        dz_rot = 0
    

    if not raw_angles:
        dx_rot = float(to_angles_0_180(dx_rot))
        dy_rot = float(to_angles_0_180(dy_rot))
        dz_rot = float(to_angles_0_180(dz_rot))
    
    return (d_x,d_y,d_z,dx_rot,dy_rot,dz_rot)

    
def twist_structure_for_scan(path_monomer_structure, path_to_structure, output_folder, ignore_bowl = True):
    
    mol_ref = read(path_monomer_structure, format='xyz',index=":")[0]
    dimer_mol = read(path_to_structure, format='xyz',index=":")[0]
    curve_list = np.array([15,8,5,4,23,27,26,22,2,3,9,10])-1

    coords = dimer_mol.get_positions()
    coords = np.array(coords)   

    curve_coords_0 = coords[curve_list,:]
    curve_coords_1 = coords[curve_list+len(mol_ref),:]
    basis_0 = get_basis_vector(curve_coords_0)
    basis_1 = get_basis_vector(curve_coords_1)
    twist_a = get_twist(dimer_mol, mol_ref, curve_list,0)
    twist_b = get_twist(dimer_mol, mol_ref, curve_list,1)
    (bowl_a1, bowl_a2) = get_bowl(dimer_mol, mol_ref, curve_list,0)
    bowl_a = np.mean(np.array([bowl_a1, bowl_a2]))
    (bowl_b1, bowl_b2) = get_bowl(dimer_mol, mol_ref, curve_list,1)

    bowl_b = np.mean(np.array([bowl_b1, bowl_b2]))*np.sign(np.dot(basis_0[2],basis_1[2]))

    (d_x,d_y,d_z,dx_rot,dy_rot,dz_rot) = get_trans_rot_dimer(dimer_mol, mol_ref, curve_list)

    twist_a_lin = np.linspace(0, 30, 31)
    twist_b_lin = np.linspace(0, 30, 31)
    twists_a, twists_b = np.meshgrid(twist_a_lin, twist_b_lin)
    twists_a = twists_a.flatten()
    twists_b = twists_b.flatten()
    
    
    for idx, (twist_a,twist_b) in enumerate(zip(twists_a,twists_b)):
        if ignore_bowl:
            attributes = [d_x,d_y,d_z,dx_rot,dy_rot,dz_rot,0,0,twist_a,twist_b]
        else:
            attributes = [d_x,d_y,d_z,dx_rot,dy_rot,dz_rot,bowl_a*2e3,-bowl_b*2e3,twist_a,twist_b]
        make_dimers_from_trans_rot(path_monomer_structure, attributes,f"{output_folder}/{idx}", comment=f"{twist_a:.1f},{twist_b:.1f}")

def do_PCA_with_diffrent_parameters(path_monomer_structure, path_dimers, path_scan_data, mode = "best_score"):

    with open("latex/PCA_tables.tex", "w") as out:
            out.writelines(["\n","\n"])

    tables = []

    n_clusters_list =    [   4,     4,     4,    4,     4,     4,    3,    3] 
    n_components_list =  [   3,     3,     3,     3,    3,     5,    3,   10]
    ignore_dz_list =     [True, True , True , True , False, True, True, True]
    ignore_rot_dx_list = [True, True , True , False, False, True, True, True]
    ignore_rot_dy_list = [True, True , False, False, False, True, True, True]
    use_absolutes_list = [True, False, False, False, False, True, True, True]

    for (n_clusters, n_components, ignore_dz, ignore_rot_dx, ignore_rot_dy, use_absolutes) in zip(n_clusters_list, n_components_list, ignore_dz_list, ignore_rot_dx_list, ignore_rot_dy_list, use_absolutes_list):

        statistic_PBI(path_monomer_structure, path_dimers, path_scan_data, mode, write_PCA=True,out_file="analysis/analysis_PCA_temp.npz",n_clusters = n_clusters, n_components = n_components, ignore_dz = ignore_dz, ignore_rot_dx = ignore_rot_dx, ignore_rot_dy = ignore_rot_dy, use_absolutes = use_absolutes)
        tables.append(latex_for_PCA_analysis(file_name="analysis/analysis_PCA_temp.npz",n_clusters = n_clusters, n_components = n_components, ignore_dz = ignore_dz, ignore_rot_dx = ignore_rot_dx, ignore_rot_dy = ignore_rot_dy, use_absolutes = use_absolutes))

    for table in tables:
        with open("latex/PCA_tables.tex", "a") as out:
            out.writelines(table)
            out.writelines(["\n","\n"])

        
if __name__ == '__main__':
    pass