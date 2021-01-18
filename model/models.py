from .fast_vth_search_model import FastVthSearch

def create_model(opt):
	model = FastVthSearch()
	model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model
