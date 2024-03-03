static void aSumHalBox01_ct(FTR_ACTOR* ftr_actor, u8* data);
static void aSumHalBox01_mv(FTR_ACTOR* ftr_actor, ACTOR* my_room_actor, GAME* game, u8* data);
static void aSumHalBox01_dw(FTR_ACTOR* ftr_actor, ACTOR* my_room_actor, GAME* game, u8* data);
static void aSumHalBox01_dt(FTR_ACTOR* ftr_actor, u8* data);

static aFTR_vtable_c aSumHalBox01_func = {
	&aSumHalBox01_ct,
	&aSumHalBox01_mv,
	&aSumHalBox01_dw,
	&aSumHalBox01_dt,
	NULL,
};

aFTR_PROFILE iam_sum_hal_box01 = {
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	NULL,
	27.0f,
	0.01f,
	aFTR_SHAPE_TYPEA,
	mCoBG_FTR_TYPEA,
	0,
	0,
	0,
	0,
	&aSumHalBox01_func,
};
