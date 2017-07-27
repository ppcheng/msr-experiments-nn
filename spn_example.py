from tachyon.SPN2 import SPN

# make an SPN holder
spn = SPN()
# include training and testing data
# spn.add_data('data/abalone/abalone.ts.data', 'train', cont=True)
# spn.add_data('data/abalone/abalone.valid.data', 'valid', cont=True)
# spn.add_data('data/abalone/abalone.test.data', 'test', cont=True)
spn.add_data('./data/Holmes_Training_Data_5gram_matrix/1ADAM10.TXT', 'train', cont=True)
# create a valid sum product network

sum_branch_factor = (2, 4)
prod_branch_factor = (20, 40)
variables = 1500

#continuous model
spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=True, classify=False, data=spn.data.train)

# start the session
spn.start_session()

# train

# pick the optimization algorithm
cccp = False
gd = True
count = False #take a step of counting then a step of gradient descent

#large minibatches with a small number at a time
minibatch_size=100

# other stuff
epochs = 1

# access the data
train = spn.data.train
valid = spn.data.valid
test = spn.data.test


spn.train(epochs, train, cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size)
# test_loss = spn.evaluate(test, minibatch_size=1)
# print 'Loss:', test_loss
# Loss: 4.513