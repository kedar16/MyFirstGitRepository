# Intermodal vs. modal transport network flexibility model

# Require packages
from graph_tool.all import *
import random
import numpy
import scipy.stats as stats
import matplotlib
from joblib import Parallel, delayed  
import multiprocessing
import time

# Create package objects
class Package(object):
	source_node = 0
	destination = 0
	current_location = 0
	itinerary = []
	mode = 0
	ID = 0
	status = "NULL"
	
	def __init__(self, source_node, destination, current_location, next_location, itinerary,  ID, status):
		self.source_node = source_node
		self.destination = destination
		self.current_location = current_location
		self.next_location = next_location
		self.itinerary = itinerary
		self.ID = ID
		self.status = status

def make_package(source_node, destination, current_location, next_location, itinerary,  ID, status):
	package = Package(source_node, destination, current_location, next_location, itinerary,  ID, status)
	return package

def find_shortest_path(source_node, destination, graph):
	vlist, elist = graph_tool.topology.shortest_path(final_graph, final_graph.vertex(source_node), final_graph.vertex(destination))		
	return vlist, elist
	#return([edge.target(), edge.source()] for edge in elist)
	
def generate_packages(pair):
	source_node = pair[0]
	destination = pair[1]
	demand = pair[2]
	graph = pair[3]
	current_location = source_node
	next_location = 0
	packages = []
	total = 0
	unmet_demand = 0

	vlist, elist = graph_tool.topology.shortest_path(graph, graph.vertex(source_node), graph.vertex(destination))		

	itinerary = [int(vertex) for vertex in vlist]
	

	if len(itinerary) == 0:
		for node_package in range(0,demand):
			packages.append(make_package(source_node, destination, current_location, next_location, itinerary, total, "unmet"))

	else:
		for node_package in range(0,demand):
			next_location = int(vlist[1])
			packages.append(make_package(source_node, destination, current_location, next_location, itinerary, total, "in_network"))

	return packages

def truncated_power_law(a, m):
#    print('a: ' + str(a))
 #   print('m: ' + str(a))
    
    x = numpy.arange(1, m+1, dtype='float')
 #   print('x: ' + str(x))
    
    pmf = 1/x**a
    pmf /= pmf.sum()
#    print('pmf: ' + str(pmf))
    meen = sum([x[i]*pmf[i] for i in range(0,len(x))])

    return stats.rv_discrete(values=(range(1, m+1), pmf))

def choose_mode(access_list, source_node, target_node, mode_degrees):
	maxmode_idx = 0
	maxmode_degree = 0
	available_modes = []
	for mode in range(0,num_modes):
		if (mode in access_list[source_node]) or (mode in access_list[target_node]):
			degree = mode_degrees[target_node,mode]
			available_modes.append([mode, degree])
			if degree > maxmode_degree:
				maxmode_degree = degree
	
	for option in available_modes:
		if option[1] < maxmode_degree:
			available_modes.remove(option)
	
	return available_modes[random.randint(0,len(available_modes)-1)][0]

def deng_li_cai_wang(pairs, access_list):
	# generate a mode-specific demand list 
	demandlist = pairs
	not_connected = 0
	mode_degrees = numpy.zeros((num_nodes, num_modes), dtype=numpy.int)



	#generate an edge list for this mode. The first edge is between the two nodes with the highest demand
	edge_list = []
	vertex_list = []
	connected_nodes = []
	max_value = numpy.argmax(demandlist)


	connected_nodes.append(max_value / num_nodes)
	connected_nodes.append(max_value % num_nodes)
	mode = choose_mode(access_list, max_value / num_nodes, max_value % num_nodes, mode_degrees)
	edge_list.append([max_value / num_nodes, max_value % num_nodes, mode])
	vertex_list.append(max_value / num_nodes)
	vertex_list.append(max_value % num_nodes)
	mode_degrees[max_value / num_nodes, mode] += 1
	mode_degrees[max_value % num_nodes, mode] += 1

	while(len(connected_nodes)<(num_nodes)): 
		#Zero-out connections between already-connected nodes
		counter = 0

		max_value = 0
		next_node = 0

		for node in connected_nodes:
			contenders = []

			for othernode in connected_nodes:
				demandlist[node,othernode] = -1 
				demandlist[othernode,node] = -1
				
			# Find the target that has the highest demand value for connection to an already-connected node
			idx = numpy.argmax(demandlist[node,:])
			maxval = numpy.max(demandlist[node,:])

			if maxval > max_value:
				next_node = idx
				max_value = maxval
			
			elif maxval == max_value:
				contenders.append(idx)

		if maxval == 0:
			next_node = contenders[numpy.random.randint(0,len(contenders))]
		
	
		#Now add the new edge, randomly choosing an existing node in the vertex list
		candidate_list = [vertex for vertex in vertex_list if vertex != next_node]
		target_node = candidate_list[numpy.random.randint(0,len(candidate_list))]
		mode = choose_mode(access_list, next_node, target_node, mode_degrees)
		edge_list.append([next_node,target_node, mode])
		mode_degrees[next_node, mode] += 1
		mode_degrees[target_node, mode] += 1
		connected_nodes.append(next_node)
		vertex_list.append(next_node)
		
		#Randomly assign m-1 edges to the first order neighbors of target
		m = 2 #I believe they meant n instead of m in the algorithm, and the initial number of connected nodes was 2. This creates a sparse graph.
		
		#make a list of all of the neighbors of target
		temp_list = []
		for edge in edge_list:
			if ( (edge[0] == target_node) and (edge[1] != next_node) ) or ( (edge[1] == target_node) and (edge[0] != next_node) ):
				counter += 1
				if edge[0] == target_node:
					mode = choose_mode(access_list, next_node, edge[1], mode_degrees)
					temp_list.append([next_node, edge[1],mode])
				else:
					mode = choose_mode(access_list, next_node, edge[0], mode_degrees)
					temp_list.append([next_node, edge[0],mode])

		edge_list.append(temp_list[numpy.random.randint(0,len(temp_list))])
		new_edge = edge_list[len(edge_list)-1]
		mode_degrees[new_edge[0], mode] += 1
		mode_degrees[new_edge[1], mode] += 1
			#do this until all nodes that can be connected are connected

	return edge_list
		
def barabasi_albert(pairs, access_list):

		
	# generate a mode-specific demand list 
	demandlist = pairs

	not_connected = 0
	
	sumdemand = sum(demandlist)
	notconnected = sum([1 for i in range(0,len(sumdemand)) if sumdemand[i] == 0])

	
	mode_degrees = numpy.zeros((num_nodes, num_modes), dtype=numpy.int)
	


		#generate an edge list for this mode. The first edge is between the two nodes with the highest demand
	edge_list = []
	vertex_list = []
	connected_nodes = []
	max_value = numpy.argmax(demandlist)
	
	source_node = max_value / num_nodes
	target_node = max_value % num_nodes
	connected_nodes.append(source_node)
	connected_nodes.append(target_node)
	vertex_list.append(source_node)
	vertex_list.append(target_node)
	

	mode = choose_mode(access_list, source_node, target_node, mode_degrees)
			
	edge_list.append([source_node, target_node, mode])
	mode_degrees[source_node, mode] += 1
	mode_degrees[target_node, mode] += 1


	while(len(connected_nodes)<(num_nodes-not_connected)): 

		#Zero-out connections between already-connected nodes
		counter = 0

		max_value = 0
		next_node = 0

		for node in connected_nodes:

			for othernode in connected_nodes:
				demandlist[node,othernode] = 0
				demandlist[othernode,node] = 0

			idx = numpy.argmax(demandlist[node,:])
			maxval = numpy.max(demandlist[node,:])

			if maxval > max_value:
				next_node = idx
				max_value = maxval

		#Now add the new edge, randomly choosing an existing node in the vertex list
		if max_value>0:
			target_node = vertex_list[numpy.random.randint(0,len(vertex_list))]


			mode = choose_mode(access_list, source_node, target_node, mode_degrees)
		
			edge_list.append([next_node,target_node, mode])
			mode_degrees[source_node, mode] += 1
			mode_degrees[target_node, mode] += 1
			connected_nodes.append(next_node)
			vertex_list.append(next_node)
			vertex_list.append(target_node)
		else:
			not_connected += 1
		
			#do this until all nodes that can be connected are connected

	
	return edge_list

def make_graph(pairs, access_list, modular, layered):
	edge_list = []
	mode_list = []
	
	#if integral, make a network that connects all nodes 
	if layered:
		edge_list = deng_li_cai_wang(pairs.copy(), access_list)
	else:
		edge_list = barabasi_albert(pairs.copy(), access_list)
	
	
	final_graph = graph_tool.Graph(directed=False)
	final_graph.add_vertex(n=num_nodes)

	return final_graph,  edge_list
			
def route_package(inputs):
	input = inputs[0]
	disrupted = inputs[1]
	g = inputs[2]
	
	output = []


	if input.status == "in_network":
			
		#if the next location is not reachable

		if not disrupted:
			#Advance the package

			if (input.next_location == input.destination):
				output = make_package(input.source_node, input.destination, input.next_location, "NULL", input.itinerary[1:len(input.itinerary)],  input.ID, "delivered")
			else:
				output = make_package(input.source_node, input.destination, input.itinerary[1], input.itinerary[2], input.itinerary[1:len(input.itinerary)],  input.ID, "in_network")
		elif layered: #reroute
			vlist, elist = graph_tool.topology.shortest_path(g, g.vertex(input.current_location), g.vertex(input.destination))
			vlist = [int(vertex) for vertex in vlist]
			if len(elist)>0:
				if vlist[1] == input.destination:
					output = make_package(input.source_node, input.destination, vlist[1], "NULL", vlist[1],  input.ID, "delivered")
				else:
					output = make_package(input.source_node, input.destination, vlist[1], vlist[2], vlist[1:len(vlist)],  input.ID, "in_network")

			else:
				output = make_package(input.source_node, input.destination, input.current_location, "NULL", vlist,  input.ID, "lost")
		else:
			output = make_package(input.source_node, input.destination, input.current_location, "NULL", input.itinerary,  input.ID, "lost")

		return output
	else:
		return input

def final_inner_loop(input):

	iter = input[0]
	A = input[1] 
	source = input[2]
	destination = input[3]
	graph = input[4]
	lhat = input[5]
	path_list = []
	likelihood_list = []

	num_nodes = A.shape[0]
	numpy.random.seed(iter)


	# 2. Start with x_1 = 1, so let c = 1 (current vertex), g = 1 (likelihood) and t = 1 (counter).
	vertex = source
	path = []
	path.append(vertex)
	loglike = 0.0
	t = 0

	# 3. Set A(:,1) = 0 to ensure that the path will not return to 1.

	A[:,vertex] = 0


	while(True):

		# 4. If A(c,n) = 0 go to step 5.
		V_prime = [i for i in range(0,num_nodes) if A[vertex,i] > 0]
		num_indices = len(V_prime)
	
		if A[vertex,destination] > 0:
			# If A(c,n) = 1 and n is the only available vertex adjacent to c set x_t+1 = n and stop
			if ( num_indices == 1):
				path.append(destination)
				path_list.append(path)
				likelihood_list.append(-loglike)
# 						if path not in path_list:
# 							path_list.append(path)
# 							likelihood_list[graph].append(-loglike)
# 							if -loglike > 100:
# 								print(loglike)
# 								raw_input()
				break
	
			#If there are other vertices adjacent to c, 
			else:
				# choose the next vertex to be n with probability lhat_t.
				rand_num = numpy.random.rand()
				comparator = numpy.exp(lhat[t])
				if rand_num<comparator:
					#If this happens, set x_t+1 = n, g = g * lhat_t and stop
					loglike += (lhat[t])
					path.append(destination)
					path_list.append(path)
					likelihood_list.append(-loglike)


# 							if path not in path_list:
# 								path_list.append(path)
# 								likelihood_list[graph].append(-loglike)
# 								if -loglike > 100:
# 									print(loglike)
# 									raw_input()
					break
			
				else:
					#Otherwise, set g = g * (1 - lhat_t), A(c,n) = 0 and continue with step 5
					loglike += numpy.log(1-numpy.exp(lhat[t]))
					A[vertex,destination] = 0

		#5. Let V' be the set of possible vertices for the next step of the path
		# (V' = { i in V | A(c,i) = 1}.) 
		
		V_prime = [i for i in range(0,num_nodes) if A[vertex,i] > 0]
		num_indices = len(V_prime)

		#If V' = {}, we do not generate a valid path so stop.
		if num_indices==0:
			break
	
		else:
			#6. Choose the next vertex in in V' of the path randomly using the uniform
			# distribution on V'. So x_t+1 = i.				
			vertex = V_prime[numpy.random.randint(num_indices)]
			path.append(vertex)
	
			#7. Set c = i, g = g/|V'|, A(:,i) = 0, and t = t+1. Return to step 4.

			loglike -= numpy.log(num_indices)
			A[:,vertex] = 0
			t += 1

	# Final Step: Lastly we estimate |X*|, using (1) and the values of g, for each valid path generated.

	if iter % 100 == 0:
		print(str(graph)+'-'+str(iter))

	return path_list, likelihood_list

def naive_inner_loop(input):


	iter = input[0]
	A = input[1] 
	source = input[2]
	destination = input[3]
	graph = input[4]


#	num_nodes = A.shape[0]
	numpy.random.seed(iter)

	#1. Start with x_1 = 1; let c = 1 (current vertex), g = 1 (likelihood) and t = 1 (counter)
	path = []
	vertex = source
	path.append(vertex)

	loglike = 0.0
	t=0

	#2. Set A(:,1) = 0 to ensure that the path will not return to 1.

	A[:,vertex] = 0

	while(True):
		#3. Let V' be the set of possible vertices for the next step of the path. 
		# (V' = {i in V | A(c,i) = 1}.) 
		V_prime = [i for i in range(0,num_nodes) if A[vertex,i] > 0]


		num_indices = len(V_prime)


		#If V' = {}, we do not generate a valid path so stop.
		if num_indices==0:
			break
		else:
			#4. Choose the next vertex i in V' of the path randomly using the uniform 
			#distribution on V'. So x_(t+1) = i
	
			vertex = V_prime[numpy.random.randint(num_indices)]
			path.append(vertex)
		
			#5. Set c=i, g = g/|V'|, A(:,i) = 0, and t = t+1
		
			loglike -= numpy.log(num_indices)
			A[:,vertex] = 0
			t += 1


			if (vertex == destination): # and (path not in path_list):				
				break

	if iter % 100 == 0:
		print('Pilot: '+ str(graph)+'-'+(str(iter)))

	return [path, loglike]	

def estimate_likelihood(adjacency_matrix, path_lists, likelihood_lists, destination, iterations):
	num_graphs = len(adjacency_matrix)
	num_nodes = adjacency_matrix[0].shape[0]
	lhat = [0]*(num_nodes-1) #A graph with 10 nodes has paths of at most 9 edges, so we iterate through nodes 1-9, indexed by 0-8.

	for graph in range(0,num_graphs):
	
	
		path_list = path_lists[graph]
		likelihood_list = likelihood_lists[graph]
		len_list = map(len,path_list)
		adj = adjacency_matrix[graph]



		numsum = 0.0
		densum = 0.0
#		print([numpy.exp(-likelihood_list[i]) for i in range(0,len(likelihood_list))])
#		raw_input()
		estimated_paths = sum([numpy.exp(-likelihood_list[i]) for i in range(0,len(likelihood_list))]) / iterations

		for k in range(0,num_nodes-1): #range(0,num_nodes-1) includes zero and excludes num_nodes-1. 
			numlist = []
			denlist = []

			# When k = 0, this indicates a path with one edge, and therefore two nodes. There are k+1 edges and k+2 nodes in every path
			numlist = [-likelihood_list[i] for i in range(0,len(path_list)) if len_list[i]==(k+2)] # Length of two means there is one edge
			denlist = [-likelihood_list[i] for i in range(0,len(path_list)) if (len_list[i]>=(k+2) and adj[path_list[i][k],destination]>0)]
	

			numsum = (sum(map(numpy.exp,numlist)))
			densum = (sum(map(numpy.exp,denlist)))
		
			if densum > 0 and numsum > 0:
				if densum == numsum:
					lhat[k] += (1- (1 / estimated_paths) )/ num_graphs
#					print('One-'+str(k)+'. ' + str(lhat[k]))
		
				else:
					lhat[k] += (numsum / densum) / num_graphs
#					print('Other-'+str(k)+'. ' +str(lhat[k]))

			elif lhat[k] == 0:
				lhat[k] += (1 / estimated_paths) / num_graphs
#				print('Zero-'+str(k)+'.  '+str(lhat[k]))
			
			
#	print(lhat)
#	raw_input()
	return(map(numpy.log,lhat))

def naive_path_generation(iterations, adjacency_matrix, path_lists, likelihood_lists, iter, source, destination):

	num_graphs = len(adjacency_matrix)
	num_cores = multiprocessing.cpu_count()
	print('Num_Cores: ' + str(num_cores))

	for graph in range(0,num_graphs):
		path_list = []
		likelihood_list = []
#		iter = 0
	
		loop_input = [ [i, adjacency_matrix[graph].copy(), source, destination, graph] for i in range(iter,iterations)] 
		inputs = range(len(loop_input))

		outputs = Parallel(n_jobs=num_cores,verbose=1)(delayed(naive_inner_loop)(li) for li in loop_input)

#		print(outputs)
		path_list = [outputs[i][0] for i in range(0,len(outputs))]
		likelihood_list = [outputs[i][1] for i in range(0,len(outputs))]

		indices = [i for i in range(0,len(path_list)) if len(path_list[i])>0]
		path_lists[graph] = [path_list[i] for i in indices]
		likelihood_lists[graph] = [likelihood_list[i] for i in indices]
#	print(len(path_lists[0]))
#	raw_input()	

	lhat = estimate_likelihood(adjacency_matrix, path_lists, likelihood_lists, destination, iterations)
	return(path_lists, likelihood_lists, lhat)

def length_distribution_method(adj_final, final_samples, lhat, estimated_paths, likelihood_lists, iter, source, destination):

	# 1. Simulate a pilot run of N' samples using the "naive" method to find an estimate, lhat
	# of the length distribution, l
#	partial_path_list, lhat = naive_path_generation(naive_samples, adj_naive, source, destination)
#	print(lhat)
#	raw_input()
	num_graphs = len(adj_final)
	num_cores = multiprocessing.cpu_count()
	likelihood_list = [0]*num_graphs
	print('Num_Cores: ' + str(num_cores))

	for graph in range(0,num_graphs):
	
		path_list = []
		likelihood_list[graph] = []

		loop_input = [ [i, adj_final[graph].copy(), source, destination, graph, lhat] for i in range(iter,final_samples)] 
		inputs = range(len(loop_input))
	
		outputs = Parallel(n_jobs=num_cores,verbose=0)(delayed(final_inner_loop)(loop_input[i]) for i in inputs)


		indices = [i for i in range(0,len(outputs)) if len(outputs[i][0])>0]


		path_list.extend(outputs[i][0][0] for i in indices)
	
		likelihood_list[graph].extend(outputs[i][1][0] for i in indices)

#		indices = [i for i in range(0,len(path_list)) if len(path_list[i])>0]
#		path_list[graph] = [path_list[i] for i in indices]
#		likelihood_list[graph] = [likelihood_list[i] for i in indices]
			
		if len(likelihood_list[graph])>0:
			likelihood_list[graph] = map(numpy.exp, likelihood_list[graph])
			estimated_paths[graph] = (sum(likelihood_list[graph]) / final_samples)
		else:
			estimated_paths[graph] = 0
		
	mu = sum(estimated_paths) / num_graphs

	l_one_hat = [0]*num_graphs
	l_two_hat = [0]*num_graphs
	relative_error = [0]*num_graphs
	relative_error = [0]*num_graphs
	for graph in range(0,num_graphs):
		relative_error[graph] = numpy.std(likelihood_list[graph]) / ((final_samples**0.5) * estimated_paths[graph])
	
	
		l_one_hat[graph] = (estimated_paths[graph] - mu)*(estimated_paths[graph] - mu)
		counter = 0
		for entry in range(0,len(likelihood_list[graph])):
			counter += 1
			l_two_hat[graph] += (likelihood_list[graph][entry] - estimated_paths[graph])*(likelihood_list[graph][entry] - estimated_paths[graph])
		l_two_hat[graph] += ((-estimated_paths[graph])*(-estimated_paths[graph])) * (final_samples - counter)
	
	
		lllist = likelihood_list[graph]
		for i in range(0,final_samples - len(likelihood_list[graph])):
			lllist.append(0.0)
		l_two_hat[graph] = numpy.std(lllist)**2
	


# 	print(l_one_hat)
# 	print(l_two_hat)
# 	raw_input()
#	lonehat = sum(l_one_hat) / (num_graphs-1)
	lonehat = numpy.std(estimated_paths)**2

#	ltwohat = sum(l_two_hat) / (num_graphs * (final_samples - 1) ) 
	ltwohat = sum(l_two_hat) / num_graphs

# 	print(lonehat)
# 	print(ltwohat)
# 	raw_input()

	var = lonehat - ltwohat

	return estimated_paths, relative_error, likelihood_lists, mu, var

def estimate_flexibility(g, target_re):

	paths_matrix = numpy.zeros((num_nodes, num_nodes))
	re_matrix = numpy.zeros((num_nodes, num_nodes))
	counter = 1
	naive_path_lists = [0]
	naive_likelihood_lists = [0]

	iter = 0
	re_flag = True
	re = 1

	for row in range(0,num_nodes-1):
		for column in range(row+1,num_nodes):
			print('Testing Row: ' +str(row) + ' Column: ' +str(column))
		
			if re_flag:
				while(re>target_re):
					final_samples = counter * 100
					naive_samples = counter * 50
					naive_path_lists, naive_likelihood_lists, lhat = naive_path_generation(naive_samples, [graph_tool.spectral.adjacency(g)], naive_path_lists, naive_likelihood_lists, iter / 2, row, column)
					estimated_paths, relative_error, likelihood_lists, mu, var = (length_distribution_method([graph_tool.spectral.adjacency(g)], final_samples, lhat, [0], [0], 0, row, column))
					print('Naive: '+ str(naive_samples) + '-Final: ' + str(final_samples) + '-Paths: '+ str(estimated_paths) + ', RE: ' + str(relative_error))
					counter *= 2
					iter = final_samples
					re = relative_error[0]
				re_flag = False
			else:
				naive_path_lists, naive_likelihood_lists, lhat = naive_path_generation(naive_samples, [graph_tool.spectral.adjacency(g)], [0], [0], 0, row, column)
				estimated_paths, relative_error, likelihood_lists, mu, var = (length_distribution_method([graph_tool.spectral.adjacency(g)], final_samples, lhat, [0], [0], 0, row, column))
				print('Naive: '+ str(naive_samples) + '-Final: ' + str(final_samples) + '-Paths: '+ str(estimated_paths) + ', RE: ' + str(relative_error))
			
			paths_matrix[row,column] = estimated_paths[0]
			paths_matrix[column,row] = paths_matrix[row,column]
			re_matrix[row,column] = relative_error[0]
			re_matrix[column,row] = re_matrix[row,column]
			print('Row: ' +str(row) + ' Column: ' +str(column)+ ' completed!')
	
	return paths_matrix, re_matrix

	
# Initialization
num_cores = multiprocessing.cpu_count()
print('Num_Cores: ' + str(num_cores))
outfile = open('flexiblity_multipliers.csv','a+')

outfile.write('Total Packages' + ',' + 'Delivered' + ',' + 'Unmet' + ',' + 'Lost' +',' + 'In Transit' +',' + 'Nodes' + ',' + 'Initial Edges' + ',' + 'Final Edges' + ','+ 'Cost' + ',' + 'Reachable' + ',' + 'Volatile?' + ','+ 'Failure Probability' + '\n')


# Specify: 
# - number of locations

kounter = 1
while(kounter < 2):

	if kounter == 0:
		modular = True
		layered = False
		volatile = True
		kounter += 1
	else:
		modular = False
		layered = False
		volatile = False
		kounter += 1
		
	for graph_iteration in range(0,10):
		random.seed(graph_iteration+10)
		numpy.random.seed(graph_iteration+10)
		num_nodes = 100 #Will be an external parameter

		# - number of modes

		num_modes = 3 #Will be an external parameter

		# - number of time steps

		time_steps = 100 #Will be an external parameter

		# - maximum nodal demand

		max_demand = 100 #Will be an external parameter
		demand_coefficient = 1.550878 #Will be an external parameter

		# - demand volatility for normal distribution

		volatility = 2 # Will be an external parameter

		# - Modular flag 
	# 	modular = True
	# 	layered = True
	# 	volatile = True

		# Establish a priori probability of link failure
 
		p_fail = 0.01 #Will be an external parameter

		# For each location, determine the total number of modes that can access it (uniform between 1 and num_modes)
		access_modes = numpy.random.randint(1,high=num_modes+1,size=num_nodes)

		# For each location, generate a list of modes that can access it (randomly select from range(0,num_modes) )
		access_list = []
		for node in range(0,num_nodes):
			modelist = []
			while len(modelist)<access_modes[node]:
				candidate = random.randint(0,num_modes-1)
				if candidate not in modelist:
					modelist.append(candidate) 
			access_list.append(modelist)


		# For each location, determine its total throughput as a randomly selected integer between 0 and max_demand squared
		d_throughput = truncated_power_law(demand_coefficient, max_demand)
		nodal_demand = d_throughput.rvs(size = num_nodes) -1 

	
		# For each location pair, determine its total demand as a random selected integer between 0 and max_demand according to a power law

		writefile = open('powerlaw.csv','w')
		pairlist = numpy.zeros((num_nodes,num_nodes))
		counter = 0
		for row in range(0,num_nodes-1):
			for column in range(row+1,num_nodes):
				pairlist[row,column] = nodal_demand[row]*nodal_demand[column]
				writefile.write(str(pairlist[row,column]) + '\n')
				counter += 1
		pairlist += pairlist.T


		# For each mode, use the Barabasi-Albert or layered algorithm to generate a scale-free network, with links added in order of demand.
		final_graph, edge_list = make_graph(numpy.copy(pairlist), access_list, modular, layered)
		edge_color_property = final_graph.new_edge_property("double")
		for edge in edge_list:
			final_graph.add_edge(edge[0],edge[1])
			edge_color_property[(edge[0],edge[1])]=edge[2]

	#	pos = graph_tool.draw.sfdp_layout(final_graph)
	#	print(graph_tool.centrality.betweenness(final_graph)[1])
		outgraph = "graph_modular_" + str(modular) + "_layered_"+str(layered)+"_volatile_"+str(volatile)+"_"+str(graph_iteration)+"_i.pdf"
		graph_tool.draw.graph_draw(final_graph, edge_color = edge_color_property, ecmap = matplotlib.cm.jet, output = outgraph)
		final_graph.save(outgraph+'.gt',fmt='gt')

		failed_edges = []

		initial_edges = 0

		for edge in edge_list:
			initial_edges += 1		
		
		cost = initial_edges
		# Initialization complete

		# For each time step

		total_packages = []
		packages_in_network = []
		inert_packages = []

		delivered = 0
		lost = 0
		total = 0
		unmet = 0
	
		disrupted_graph = final_graph.copy()
		disrupted_graph.set_fast_edge_removal(fast=True)
		edge_color_property = disrupted_graph.new_edge_property("double")
		
		for timestamp in range(0,time_steps):
			print(str(kounter) +'-' + str(graph_iteration) + '-' + str(timestamp))
			print('Calculating new demand...')
			t = time.time()
			if volatile:
				d = truncated_power_law(demand_coefficient, max_demand)
				new_demand = d.rvs(size = num_nodes*(num_nodes-1) / 2) - 1
			else:
				new_demand = numpy.absolute(numpy.random.normal(0,volatility, size = num_nodes*(num_nodes-1) / 2).astype(int))

			new_demand_list = numpy.zeros((num_nodes,num_nodes))
			counter = 0
			elapsed = time.time() - t
			print('Done!: ' + str(elapsed))
			print('Adjusting demand...')
			t = time.time()
			for row in range(0,num_nodes-1):
				for column in range(row+1,num_nodes):
					if(random.random() <0.5):
						new_demand_list[row,column] = new_demand[counter]
					else:
						new_demand_list[row,column] = -min(pairlist[row,column]-1,new_demand[counter])
					counter += 1
			new_demand_list += new_demand_list.T
			pairlist += new_demand_list
			elapsed = time.time() - t
			print('Done!: ' + str(elapsed))

		
		# Generate packages for each node
			print('Generating new packages...')
			t = time.time()
			new_packages = pairlist.astype(int)
	
			newly_failed_edges = []
			SDpair = [[i,j,new_packages[i,j],disrupted_graph] for i in range(0,num_nodes) for j in range(0,num_nodes)]
			new_package_list = Parallel(n_jobs=num_cores,backend='multiprocessing',verbose=1)(delayed(generate_packages)(pair) for pair in SDpair)
			elapsed = time.time() - t
			print('Done!: ' + str(elapsed))
		

			print('Disrupting graph...')
			t = time.time()
			edges_to_remove = []
			for edge in edge_list:
				if(random.random()<p_fail):
					newly_failed_edges.append(edge)
					edge_list.remove(edge)
				
					for graph_edge in disrupted_graph.edges():

						if (graph_edge.source() == edge[0] and graph_edge.target() == edge[1]) or (graph_edge.source() == edge[1] and graph_edge.target() == edge[0]):
							edges_to_remove.append(graph_edge)


			if len(edges_to_remove)>0:
				[disrupted_graph.remove_edge(edges_to_remove[i]) for i in range(0,len(edges_to_remove)) if edges_to_remove[i] in disrupted_graph.edges()]

			failed_edges.extend(newly_failed_edges)
			elapsed = (time.time() - t)
			print('Done!: ' + str(elapsed))

			print('Queuing packages...')
			t = time.time()
		
			inputs = []
	
			for i in range(0,len(new_package_list)):
				if len(new_package_list[i])>0:
					packages_in_network.extend(new_package_list[i])
			elapsed = (time.time() - t)
			print('Done!: ' + str(elapsed))
			print('Readying packages for transport ...')

			t = time.time()

			for pkg in packages_in_network: #outputs[i][0]:
				if pkg.status != "in_network":
					inert_packages.append(pkg)
				else:
					if disrupted_graph.edge(pkg.current_location,pkg.next_location): #if the edge exists
						disrupted = False
					else:
						disrupted = True
					inputs.append([pkg,disrupted,disrupted_graph])
			elapsed=time.time()-t
			print('Done!: ' + str(elapsed))
			print('Routing packages ...')
			t = time.time()

			packages_in_network = Parallel(n_jobs=num_cores,backend='multiprocessing',verbose=1)(delayed(route_package)(input) for input in inputs)
			elapsed=time.time()-t
			print('Done!: ' + str(elapsed))

			print('Consolidating packages ...')
			t = time.time()

			total_packages = inert_packages + packages_in_network 
			elapsed=time.time()-t
			print('Done!: ' + str(elapsed))
		
			# Try to add new edges if possible
		
			if modular:
				#if there is a failed edge, try to replace it
				print('Attempting to replace failed edges  ...')
				t = time.time()

				for edge in newly_failed_edges:
					candidate_edges = [[edge[0],edge[1],mode] for mode in range(0,num_modes) if (mode in access_list[edge[0]] and mode in access_list[edge[1]])]
					candidate_edges = [edge for edge in candidate_edges if edge not in failed_edges]
					if len(candidate_edges)>0:
						edge = candidate_edges[numpy.random.randint(0,len(candidate_edges))]
					disrupted_graph.add_edge(edge[0],edge[1])
					edge_list.append(edge)
					cost += 1
				elapsed=time.time()-t
				print('Done!: ' + str(elapsed))


				#if there is unmet demand, try to fill it
				print('Attempting to fill unmet demand  ...')
				t = time.time()
				unmet_list = [[total_packages[i].source_node, total_packages[i].destination] for i in range(0,len(total_packages)) if total_packages[i].status == "unmet"]
			
				u_list = []
				for edge in unmet_list:
					if [edge[0], edge[1]] not in u_list and [edge[1], edge[0]] not in u_list:
						u_list.append(edge)

	#			unconnected_vertices = [int(vertex) for vertex in disrupted_graph.vertices() if vertex.out_degree() == 0]
	#			print(unconnected_vertices)
			
				for edge in u_list:
				#In the next iteration of the algorithm, this should ask if edge[0] is reachable from edge[1]; not if either is disconnected. This has to happen serially. 			
					vlist, elist = graph_tool.topology.shortest_path(disrupted_graph, disrupted_graph.vertex(edge[0]), disrupted_graph.vertex(edge[1]))
					if (len(elist) == 0): #if the two vertices are still unreachable from one another
						#add an edge 
						candidate_edges = [[edge[0],edge[1],mode] for mode in range(0,num_modes) if ((edge not in failed_edges) and (mode in access_list[edge[0]] and mode in access_list[edge[1]]))]
						if len(candidate_edges)>0:
								candidate_edge = candidate_edges[numpy.random.randint(0,len(candidate_edges))]
								disrupted_graph.add_edge(candidate_edge[0],candidate_edge[1])
	#						unconnected_vertices = [vertex for vertex in unconnected_vertices if vertex not in candidate_edge]
								edge_list.append(candidate_edge)
								cost += 1
								elapsed=time.time()-t
								print('Done!: ' + str(elapsed))

			print('Reporting statistics  ...')
			t = time.time()
			in_transit = 0
			for pkg in total_packages:
	
				sts = pkg.status
				if pkg.status == "in_network":
					in_transit += 1
				else:

					if sts == "unmet":
						unmet += 1
					
					elif sts == "delivered":
						delivered += 1
					elif sts == "lost":
						lost += 1
			total = unmet + delivered + lost + in_transit
			print('Total Demand: ' + str(total))
			print('Unmet Demand: ' + str(unmet))
			print('Total Delivered: ' + str(delivered))
			print('Packages Lost: ' +str(lost))
			print('In Transit: ' + str(in_transit))
			print('Cost: ' + str(cost))
			elapsed=time.time()-t
			print('Done!: ' + str(elapsed))
		
			print('Cleaning up memory...')
			t = time.time()
			total_packages = []
			new_package_list = []
			newly_failed_edges = []
			SDpair = []
			edges_to_remove = []
			inert_packages = []
			inputs = []
			candidate_edges = []
			unmet_list = []
			u_list = []
			elapsed=time.time()-t
			print('Done!: ' + str(elapsed))
	#	total = len(total_packages)
		print('Total Demand: ' + str(total))
	
	#	unmet = sum([1 for i in range(0,len(total_packages)) if total_packages[i].status == "unmet"])
		print('Unmet Demand: ' + str(unmet))

	#	delivered = sum([1 for i in range(0,len(total_packages)) if total_packages[i].status == "delivered"])
		print('Total Delivered: ' + str(delivered))

	#	lost = sum([1 for i in range(0,len(total_packages)) if total_packages[i].status == "lost"])
		print('Packages Lost: ' +str(lost))

	#	in_transit = sum([1 for i in range(0,len(total_packages)) if total_packages[i].status == "in_network"])
		print('In Transit: ' + str(in_transit))

		print('Cost: ' + str(cost))


	# 		for edge in edge_list:
	# 			edge_color_property[(edge[0],edge[1])]=edge[2]		
	# 		graph_tool.draw.graph_draw(disrupted_graph, edge_color = edge_color_property, ecmap = matplotlib.cm.jet, output="test.pdf")

		final_edges = 0
		for edge in disrupted_graph.edges():
			final_edges += 1

		num_reachable = 0 
		for vertex in final_graph.vertices():
			if vertex.out_degree > 0:
				num_reachable += 1
		outfile.write(str(total) + ',' + str(delivered) + ',' + str(unmet) + ',' + str(lost)+',' + str(in_transit)+',' + str(num_nodes) + ',' + str(initial_edges) + ',' + str(final_edges) + ',' + str(cost) + ',' + str(num_reachable) + ',' +str(volatile) + ','+ str(p_fail) + '\n')

		edge_color_property = disrupted_graph.new_edge_property("double")
		for edge in edge_list:
	#		disrupted_graph.add_edge(edge[0],edge[1])
			edge_color_property[(edge[0],edge[1])]=edge[2]

	#	pos = graph_tool.draw.sfdp_layout(final_graph)
	#	print(graph_tool.centrality.betweenness(final_graph)[1])
		outgraph = "graph_modular_" + str(modular) + "_layered_"+str(layered)+"_volatile_"+str(volatile)+"_"+str(graph_iteration)+"_f.pdf"
		disrupted_graph.save(outgraph+'.gt',fmt='gt')
		graph_tool.draw.graph_draw(disrupted_graph, edge_color = edge_color_property, ecmap = matplotlib.cm.jet, output=outgraph)
	outfile.close()
	#flexibility_paths, relative_error = estimate_flexibility(final_graph.copy(), 0.05)
	#print(flexibility_paths)
	#print(relative_error)
# root@kraken04:/tmp/jason# 
