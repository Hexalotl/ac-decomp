extern Gfx int_ike_pst_pig01_on_model[];
extern Gfx int_ike_pst_pig01_onT_model[];
static void fIPP_mv(FTR_ACTOR* ftr_actor, ACTOR* my_room_actor, GAME* game, u8* data);

static aFTR_vtable_c fIPP_func = {
	NULL,
	&fIPP_mv,
	NULL,
	NULL,
	NULL,
};

aFTR_PROFILE iam_ike_pst_pig01 = {
	int_ike_pst_pig01_on_model,
	int_ike_pst_pig01_onT_model,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	15.7f,
	0.01f,
	aFTR_SHAPE_TYPEA,
	mCoBG_FTR_TYPEA,
	0,
	0,
	0,
	0,
	&fIPP_func,
};
