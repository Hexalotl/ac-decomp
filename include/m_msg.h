#ifndef M_MSG_H
#define M_MSG_H

#include "types.h"
#include "libu64/gfxprint.h"
#include "m_choice.h"
#include "m_item_name.h"
#include "m_msg_data.h"

#ifdef __cplusplus
extern "C" {
#endif

#define mMsg_MSG_BUF_SIZE 1600
#define mMsg_FREE_STRING_LEN 16
#define mMsg_MAIL_STRING_LEN 132

enum {
  mMsg_INDEX_HIDE,
  mMsg_INDEX_APPEAR,
  mMsg_INDEX_NORMAL,
  mMsg_INDEX_CURSOL,
  mMsg_INDEX_DISAPPEAR,
  mMsg_INDEX_APPEAR_WAIT,
  mMsg_INDEX_WAIT,
  mMsg_INDEX_DISAPPEAR_WAIT,

  mMsg_INDEX_NUM
};

enum {
  mMsg_FREE_STR0,
  mMsg_FREE_STR1,
  mMsg_FREE_STR2,
  mMsg_FREE_STR3,
  mMsg_FREE_STR4,
  mMsg_FREE_STR5,
  mMsg_FREE_STR6,
  mMsg_FREE_STR7,
  mMsg_FREE_STR8,
  mMsg_FREE_STR9,
  mMsg_FREE_STR10,
  mMsg_FREE_STR11,
  mMsg_FREE_STR12,
  mMsg_FREE_STR13,
  mMsg_FREE_STR14,
  mMsg_FREE_STR15,
  mMsg_FREE_STR16,
  mMsg_FREE_STR17,
  mMsg_FREE_STR18,
  mMsg_FREE_STR19,

  mMsg_FREE_STR_NUM
};

enum {
  mMsg_ITEM_STR0,
  mMsg_ITEM_STR1,
  mMsg_ITEM_STR2,
  mMsg_ITEM_STR3,
  mMsg_ITEM_STR4,

  mMsg_ITEM_STR_NUM
};

enum {
  mMsg_MAIL_STR0,

  mMsg_MAIL_STR_NUM
};

#define mMsg_STATUS_FLAG_ZOOMDOWN_LONG (1 << 11) /* When set, mMsg_sound_ZOOMDOWN_SHORT() sfx will not play */
#define mMsg_STATUS_FLAG_USE_AM (1 << 17) /* 'AM' when set, 'PM' when not set */

typedef struct message_window_s mMsg_Window_c;
typedef struct message_data_s mMsg_Data_c;

/* These are hinted at due to the 8-byte alignment forced onto these structs */

typedef struct {
  int saved_main_index;
} mMsg_MainAppearWait_Data_c;

typedef struct {
  int saved_main_index;
} mMsg_MainDisappearWait_Data_c;

typedef struct {
  int saved_main_index;
} mMsg_MainWait_Data_c;

typedef union {
  mMsg_MainAppearWait_Data_c main_appear_wait;
  mMsg_MainDisappearWait_Data_c main_disappear_wait;
  mMsg_MainWait_Data_c main_wait;
  u64 align;
} mMsg_Main_Data_c;

typedef struct {
  ACTOR* speaker_actor;
  int msg_no;
  int name_shown_flag;
  rgba_t window_color;
} mMsg_Request_MainAppear_Data_c;

typedef struct {
  int saved_main_index;
  int init_flags;
} mMsg_Request_MainAppearWait_Data_c;

typedef struct {
  int init_flags;
} mMsg_Request_MainCursor_Data_c;

typedef struct {
  int init_flags;
} mMsg_Request_MainDisappearWait_Data_c;

typedef struct {
  int init_flags;
} mMsg_Request_MainNormal_Data_c;

typedef struct {
  int saved_main_index;
} mMsg_Request_MainWait_Data_c;

typedef union {
  mMsg_Request_MainAppear_Data_c request_main_appear;
  mMsg_Request_MainAppearWait_Data_c request_main_appear_wait;
  mMsg_Request_MainCursor_Data_c request_main_cursor;
  mMsg_Request_MainDisappearWait_Data_c request_main_disappear_wait;
  mMsg_Request_MainNormal_Data_c request_main_normal;
  mMsg_Request_MainWait_Data_c request_main_wait;
  u64 align;
} mMsg_Request_Data_c;

typedef union {
  u8 data[mMsg_MSG_BUF_SIZE];
  u64 align;
} mMsg_MsgBuf_c;

struct message_data_s {
  int data_loaded;
  int msg_no;
  int msg_len;
  int cut;

  mMsg_MsgBuf_c text_buf;
};

struct message_window_s {
  /* 0x000 */ int data_loaded;
  /* 0x004 */ int msg_no;
  /* 0x008 */ int _008;
  /* 0x00C */ mMsg_Data_c* msg_data;
  /* 0x010 */ f32 center_x;
  /* 0x014 */ f32 center_y;
  /* 0x018 */ f32 width;
  /* 0x01C */ f32 height;
  
  /* 0x020 */ ACTOR* talk_actor;
  /* 0x024 */ int show_actor_name;
  /* 0x028 */ int actor_name_len;
  /* 0x02C */ int nameplate_x;
  /* 0x030 */ int nameplay_y;

  /* 0x034 */ int show_continue_button;

  /* 0x038 */ u8 free_str[mMsg_FREE_STR_NUM][mMsg_FREE_STRING_LEN];
  /* 0x178 */ int free_str_article[mMsg_FREE_STR_NUM];

  /* 0x1C8 */ u8 item_str[mMsg_ITEM_STR_NUM][mMsg_FREE_STRING_LEN];
  /* 0x218 */ int item_str_article[mMsg_ITEM_STR_NUM];

  /* 0x22C */ u8 mail_str[mMsg_MAIL_STR_NUM][mMsg_MAIL_STRING_LEN];

  /* 0x2B0 */ rgba_t name_text_color;
  /* 0x2B4 */ rgba_t name_background_color;

  /* 0x2B8 */ rgba_t window_background_color;
  /* 0x2BC */ rgba_t font_color[4];

  /* 0x2CC */ rgba_t continue_button_color;

  /* 0x2D0 */ f32 font_scale_x;
  /* 0x2D4 */ f32 font_scale_y;

  /* 0x2D8 */ int _2D8;
  /* 0x2DC */ int _2DC;

  /* 0x2E0 */ int text_lines;
  /* 0x2E4 */ int current_line;

  /* 0x2E8 */ mChoice_c choice_window;

  /* 0x3E8 */ int _3E8;

  /* 0x3EC */ u16 end_timer;
  /* 0x3EE */ s16 animal_voice_idx;
  /* 0x3F0 */ int voice_sfx_idx;
  /* 0x3F4 */ u8 voice_idx;
  /* 0x3F5 */ u8 voice2_idx;
  /* 0x3F6 */ u8 voice3_idx;
  /* 0x3F7 */ s8 hide_choice_window_timer;
  /* 0x3F8 */ u8 force_voice_enable_flag;
  /* 0x3FC */ int spec;
  /* 0x400 */ u8 free_str_color_idx[4];
  /* 0x408 */ u8 _404[8]; // unused?
  /* 0x40C */ u32 status_flags;

  /* 0x410 */ f32 timer;
  /* 0x414 */ f32 cursor_timer;
  /* 0x418 */ f32 continue_button_timer;

  /* 0x41C */ int start_text_cursor_idx;
  /* 0x420 */ int end_text_cursor_idx;
  /* 0x424 */ f32 window_scale;
  /* 0x428 */ f32 text_scale;

  /* 0x42C */ int requested_main_index;
  /* 0x430 */ int requested_priority;

  /* 0x434 */ int main_index;
  /* 0x438 */ int draw_flag;
  /* 0x43C */ int cancel_flag;
  /* 0x440 */ int cancelable_flag;
  /* 0x444 */ int continue_msg_no;
  /* 0x448 */ int continue_cancel_flag;
  /* 0x44C */ int force_next;
  /* 0x450 */ int lock_continue;
  /* 0x454 */ s8 now_utter;

  /* 0x458 */ mMsg_Main_Data_c main_data;
  /* 0x460 */ mMsg_Request_Data_c request_data;
};

extern int mMsg_Get_Length_String(u8* buf, size_t buf_size);
extern mMsg_Window_c* mMsg_Get_base_window_p();
extern void mMsg_Set_free_str(mMsg_Window_c* msg, int free_str_no, u8* str, int str_size);
extern void mMsg_Set_free_str_art(mMsg_Window_c* msg, int free_str_no, u8* str, int str_size, int article_no);
extern void mMsg_Set_item_str_art(mMsg_Window_c* msg, int free_str_no, u8* str, int str_size, int article_no);
extern void mMsg_ct(GAME_PLAY*);
extern void mMsg_dt(GAME_PLAY*);
extern void mMsg_Main(GAME_PLAY*);
extern void mMsg_Draw(GAME_PLAY*);
extern void mMsg_debug_draw(gfxprint_t* gfxprint);
extern void mMsg_aram_init();
extern void mMsg_aram_init2();
extern int mMsg_Check_MainHide(mMsg_Window_c* msg);
extern void mMsg_Set_item_str(mMsg_Window_c* msg_win, int str_no, u8* item_str, int str_len);
extern void mMsg_Set_mail_str(mMsg_Window_c* msg_win, int str_no, u8* str, int str_len);
extern void mMsg_Set_continue_msg_num(mMsg_Window_c* msg_win, int continue_msg_num);
extern int mMsg_Check_MainNormalContinue(mMsg_Window_c* msg_win);
extern int mMsg_Check_main_wait(mMsg_Window_c* msg_win);
extern int mMsg_ChangeMsgData(mMsg_Window_c* msg_win, int msg_no);
extern void mMsg_Set_ForceNext(mMsg_Window_c* msg_win);
extern int mMsg_Check_not_series_main_wait(mMsg_Window_c* msg_win);
extern int mMsg_Check_MainDisappear(mMsg_Window_c* msg_win);
extern void mMsg_request_main_disappear_wait_type1(mMsg_Window_c* msg_win);
extern void mMsg_request_main_appear_wait_type1(mMsg_Window_c* msg_win);
extern void mMsg_Get_BodyParam(u32 table_rom_start, u32 data_rom_start, int entry_no, u32* data_addr, u32* data_size);
extern void mMsg_Set_LockContinue(mMsg_Window_c* msg_win);
extern void mMsg_Unset_LockContinue(mMsg_Window_c* msg_win);
extern int mMsg_Get_msg_num(mMsg_Window_c* msg_win);
extern int mMsg_Check_main_index(mMsg_Window_c* msg_win, int index);
extern int mMsg_request_main_appear(mMsg_Window_c* msg_win, ACTOR* other_actor, int display_name, rgba_t* window_color_p, int msg_no, int request_priority);
extern int mMsg_Check_main_hide(mMsg_Window_c* msg_win);
extern int mMsg_sound_voice_get_for_editor(int code);
extern int mMsg_sound_spec_change_voice(mMsg_Window_c* msg_win);
extern void mMsg_request_main_forceoff();
extern int mMsg_CopyPlayerName(u8* data, int idx, int max_size, int capitalize);
extern int mMsg_CopyTalkName(ACTOR* actor, u8* data, int idx, int max_size, int capitalize);
extern int mMsg_CopyTail(ACTOR* actor, u8* data, int idx, int max_size, int capitalize);
extern int mMsg_CopyYear(u8* data, int idx, int max_size);
extern int mMsg_CopyMonth(u8* data, int idx, int max_size);
extern int mMsg_CopyWeek(u8* data, int idx, int max_size);
extern int mMsg_CopyDay(u8* data, int idx, int max_size);
extern int mMsg_CopyHour(u8* data, int idx, int max_size);
extern int mMsg_CopyMin(u8* data, int idx, int max_size);
extern int mMsg_CopySec(u8* data, int idx, int max_size);
extern int mMsg_CopyFree(mMsg_Window_c* msg_win, int free_idx, u8* data, int idx, int max_size, int article, int capitalize);
extern int mMsg_CopyDetermination(mMsg_Window_c* msg_win, u8* data, int idx, int max_size);
extern int mMsg_CopyCountryName(u8* data, int idx, int max_size, int capitalize);
extern int mMsg_CopyRamdomNumber2(u8* data, int idx, int max_size);
extern int mMsg_CopyItem(mMsg_Window_c* msg_win, int item_idx, u8* data, int idx, int max_size, int article, int capitalize);
extern int mMsg_CopyMail(mMsg_Window_c* msg_win, int mail_idx, u8* data, int idx, int max_size);
extern int mMsg_CopyIslandName(u8* data, int idx, int max_size, int capitalize);
extern int mMsg_CopyAmPm(mMsg_Window_c* msg_win, u8* data, int idx, int max_size);
extern void mMsg_sound_set_voice_silent(mMsg_Window_c* msg_win, int update_voice_mode);
extern void mMsg_sound_unset_voice_silent(mMsg_Window_c* msg_win, int update_voice_mode);

#ifdef __cplusplus
}
#endif

#endif
