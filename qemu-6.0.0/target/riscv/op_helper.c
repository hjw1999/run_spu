/*
 * RISC-V Emulation Helpers for QEMU.
 *
 * Copyright (c) 2016-2017 Sagar Karandikar, sagark@eecs.berkeley.edu
 * Copyright (c) 2017-2018 SiFive, Inc.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2 or later, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "cpu.h"
#include "qemu/main-loop.h"
#include "exec/exec-all.h"
#include "exec/helper-proto.h"

extern Barrier barrier[BARRIER_NUM]; //for barrier

/* Exceptions processing helpers */
void QEMU_NORETURN riscv_raise_exception(CPURISCVState *env,
                                          uint32_t exception, uintptr_t pc)
{
    npu_log("riscv_raise_exception: pc=%lx expt=%d\n\r", pc, exception);
    CPUState *cs = env_cpu(env);
    cs->exception_index = exception;
    cpu_loop_exit_restore(cs, pc);
}

void helper_raise_exception(CPURISCVState *env, uint32_t exception)
{
    // npu_log("helper_raise_exception: excp=%d GETPC=%lx env->pc=%x\n\r", exception, GETPC(), env->pc);
    riscv_raise_exception(env, exception, 0);
}

target_ulong helper_csrrw(CPURISCVState *env, target_ulong src,
        target_ulong csr)
{
    target_ulong val = 0;
    int ret = riscv_csrrw(env, csr, &val, src, -1);

    if (ret < 0) {
        riscv_raise_exception(env, -ret, GETPC());
    }
    return val;
}

target_ulong helper_csrrs(CPURISCVState *env, target_ulong src,
        target_ulong csr, target_ulong rs1_pass)
{
    target_ulong val = 0;
    int ret = riscv_csrrw(env, csr, &val, -1, rs1_pass ? src : 0);

    if (ret < 0) {
        riscv_raise_exception(env, -ret, GETPC());
    }
    return val;
}

target_ulong helper_csrrc(CPURISCVState *env, target_ulong src,
        target_ulong csr, target_ulong rs1_pass)
{
    target_ulong val = 0;
    int ret = riscv_csrrw(env, csr, &val, 0, rs1_pass ? src : 0);

    if (ret < 0) {
        riscv_raise_exception(env, -ret, GETPC());
    }
    return val;
}

#ifndef CONFIG_USER_ONLY

target_ulong helper_sret(CPURISCVState *env, target_ulong cpu_pc_deb)
{
    uint64_t mstatus;
    target_ulong prev_priv, prev_virt;

    if (!(env->priv >= PRV_S)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    target_ulong retpc = env->sepc;
    if (!riscv_has_ext(env, RVC) && (retpc & 0x3)) {
        riscv_raise_exception(env, RISCV_EXCP_INST_ADDR_MIS, GETPC());
    }

    if (get_field(env->mstatus, MSTATUS_TSR) && !(env->priv >= PRV_M)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    if (riscv_has_ext(env, RVH) && riscv_cpu_virt_enabled(env) &&
        get_field(env->hstatus, HSTATUS_VTSR)) {
        riscv_raise_exception(env, RISCV_EXCP_VIRT_INSTRUCTION_FAULT, GETPC());
    }

    mstatus = env->mstatus;

    if (riscv_has_ext(env, RVH) && !riscv_cpu_virt_enabled(env)) {
        /* We support Hypervisor extensions and virtulisation is disabled */
        target_ulong hstatus = env->hstatus;

        prev_priv = get_field(mstatus, MSTATUS_SPP);
        prev_virt = get_field(hstatus, HSTATUS_SPV);

        hstatus = set_field(hstatus, HSTATUS_SPV, 0);
        mstatus = set_field(mstatus, MSTATUS_SPP, 0);
        mstatus = set_field(mstatus, SSTATUS_SIE,
                            get_field(mstatus, SSTATUS_SPIE));
        mstatus = set_field(mstatus, SSTATUS_SPIE, 1);

        env->mstatus = mstatus;
        env->hstatus = hstatus;

        if (prev_virt) {
            riscv_cpu_swap_hypervisor_regs(env);
        }

        riscv_cpu_set_virt_enabled(env, prev_virt);
    } else {
        prev_priv = get_field(mstatus, MSTATUS_SPP);

        mstatus = set_field(mstatus, MSTATUS_SIE,
                            get_field(mstatus, MSTATUS_SPIE));
        mstatus = set_field(mstatus, MSTATUS_SPIE, 1);
        mstatus = set_field(mstatus, MSTATUS_SPP, PRV_U);
        env->mstatus = mstatus;
    }

    riscv_cpu_set_mode(env, prev_priv);

    return retpc;
}

target_ulong helper_mret(CPURISCVState *env, target_ulong cpu_pc_deb)
{
    if (!(env->priv >= PRV_M)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    target_ulong retpc = env->mepc;
    if (!riscv_has_ext(env, RVC) && (retpc & 0x3)) {
        riscv_raise_exception(env, RISCV_EXCP_INST_ADDR_MIS, GETPC());
    }

    uint64_t mstatus = env->mstatus;
    target_ulong prev_priv = get_field(mstatus, MSTATUS_MPP);

    if (!pmp_get_num_rules(env) && (prev_priv != PRV_M)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    target_ulong prev_virt = get_field(env->mstatus, MSTATUS_MPV);
    mstatus = set_field(mstatus, MSTATUS_MIE,
                        get_field(mstatus, MSTATUS_MPIE));
    mstatus = set_field(mstatus, MSTATUS_MPIE, 1);
    mstatus = set_field(mstatus, MSTATUS_MPP, PRV_U);
    mstatus = set_field(mstatus, MSTATUS_MPV, 0);
    env->mstatus = mstatus;
    riscv_cpu_set_mode(env, prev_priv);

    if (riscv_has_ext(env, RVH)) {
        if (prev_virt) {
            riscv_cpu_swap_hypervisor_regs(env);
        }

        riscv_cpu_set_virt_enabled(env, prev_virt);
    }

    return retpc;
}

void helper_wfi(CPURISCVState *env)
{
    printf("======wfi=====(PC: %x)\n\r", env->pc);
    CPUState *cs = env_cpu(env);

    if ((env->priv == PRV_S &&
        get_field(env->mstatus, MSTATUS_TW)) ||
        riscv_cpu_virt_enabled(env)) {
        riscv_raise_exception(env, RISCV_EXCP_VIRT_INSTRUCTION_FAULT, GETPC());
    } else {
        cs->halted = 1;
        cs->exception_index = EXCP_HLT;
        cpu_loop_exit(cs);
    }
}

void helper_barrier(CPURISCVState *env, uint32_t rs) // for barrier
{
    uint32_t barrier_id, sync_count, hart_id;
    CPUState *cs = env_cpu(env);
    barrier_id = rs & 0xfff;
    sync_count = rs >> 12;
    hart_id = env->mhartid;
    printf("======barrier=====, hart_id: %x\n\r", hart_id);
    cs->wait_for_barrier = 1;
    cs->exception_index = EXCP_BARRIER;
    cs->barrier_id = barrier_id;
    cs->sync_count = sync_count;
    cs->hart_id = hart_id;
    // barrier
    // if(barrier[barrier_id].initialized == 0){
    //     barrier[barrier_id].initialized = 1;
    //     barrier[barrier_id].counter = sync_count;
    //     barrier[barrier_id].sync_count = sync_count;
    // }else{
    //     if(sync_count != barrier[barrier_id].sync_count){ //exit

    //     }
    //     barrier[barrier_id].counter --;
    // }
    // barrier[barrier_id].core[hart_id] = 1;
    cpu_restore_state(cs, GETPC(), true);// to do
    cpu_loop_exit(cs);
    
}

void helper_tlb_flush(CPURISCVState *env)
{
    CPUState *cs = env_cpu(env);
    if (!(env->priv >= PRV_S) ||
        (env->priv == PRV_S &&
         get_field(env->mstatus, MSTATUS_TVM))) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    } else if (riscv_has_ext(env, RVH) && riscv_cpu_virt_enabled(env) &&
               get_field(env->hstatus, HSTATUS_VTVM)) {
        riscv_raise_exception(env, RISCV_EXCP_VIRT_INSTRUCTION_FAULT, GETPC());
    } else {
        tlb_flush(cs);
    }
}

void helper_hyp_tlb_flush(CPURISCVState *env)
{
    CPUState *cs = env_cpu(env);

    if (env->priv == PRV_S && riscv_cpu_virt_enabled(env)) {
        riscv_raise_exception(env, RISCV_EXCP_VIRT_INSTRUCTION_FAULT, GETPC());
    }

    if (env->priv == PRV_M ||
        (env->priv == PRV_S && !riscv_cpu_virt_enabled(env))) {
        tlb_flush(cs);
        return;
    }

    riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
}

void helper_hyp_gvma_tlb_flush(CPURISCVState *env)
{
    if (env->priv == PRV_S && !riscv_cpu_virt_enabled(env) &&
        get_field(env->mstatus, MSTATUS_TVM)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    helper_hyp_tlb_flush(env);
}

target_ulong helper_hyp_hlvx_hu(CPURISCVState *env, target_ulong address)
{
    int mmu_idx = cpu_mmu_index(env, true) | TB_FLAGS_PRIV_HYP_ACCESS_MASK;

    return cpu_lduw_mmuidx_ra(env, address, mmu_idx, GETPC());
}

target_ulong helper_hyp_hlvx_wu(CPURISCVState *env, target_ulong address)
{
    int mmu_idx = cpu_mmu_index(env, true) | TB_FLAGS_PRIV_HYP_ACCESS_MASK;

    return cpu_ldl_mmuidx_ra(env, address, mmu_idx, GETPC());
}

#endif /* !CONFIG_USER_ONLY */

#ifdef CONFIG_NPU

#define npu_raise_exception()  {riscv_raise_exception(env, RISCV_EXCP_NPU_COMPUTE_FAULT, GETPC());}

static void vloop_clear(vloop_state_t *vstate){
    vstate->enabled = false;
    vstate->loop_num = 0;
    vstate->loop_tail_ptr = 0;
    for(int i=0; i<VEC_CMDQ_DEPTH; i++){
        vstate->records[i].type = VLOOP_MISC;
        vstate->records[i].ldst_addr = 0;
        vstate->records[i].func = NULL;
        memset(&vstate->records[i].args_list, 0, sizeof(vloop_args_t));
    }
}

static void vloop_setup(vloop_state_t *vstate, uint32_t repeat){
    vloop_clear(vstate);
    vstate->enabled = 1;
    vstate->loop_num = repeat;
}

static void vloop_finish(vloop_state_t *vstate){
    vstate->enabled = 0;
}

static vloop_args_t *vloop_push_record(vloop_state_t *vstate, vloop_insn_e type, void (*func)(uint32_t*, uint32_t), uint32_t ldst_addr){
    if(vstate->loop_tail_ptr >= VEC_CMDQ_DEPTH){
        npu_log("VLOOP PUSH ERROR\n\r");
        exit(-1);
    }
    npu_log("VLOOP RECORD #%d\n\r", vstate->loop_tail_ptr);
    vloop_entry_t *new_record = &vstate->records[vstate->loop_tail_ptr];
    vloop_args_t *args = &new_record->args_list;
    new_record->type = type;
    new_record->ldst_addr = ldst_addr;
    new_record->func = func;
    
    vstate->loop_tail_ptr++;

    return args;
}

static void vloop_push_args(vloop_args_t *arg_list, uint32_t value){
    if(arg_list->arg_tail_ptr >= VEC_ARGS_LIMIT){
        npu_log("VLOOP PUSH ERROR\n\r");
        exit(-1);
    }

    arg_list->args[arg_list->arg_tail_ptr] = value;
    arg_list->arg_tail_ptr++;
}

static void vloop_exec_once(vloop_state_t *vstate, int32_t ld_diff, int32_t st_diff){
    vloop_entry_t *record;
    uint32_t current_ldst_addr;
    for(int i=0; i<= vstate->loop_tail_ptr-1; i++){
        record = &vstate->records[i];
        if(record->type == VLOOP_LD){
            current_ldst_addr = record->ldst_addr + ld_diff;   
        }
        else if(record->type ==VLOOP_ST){
            current_ldst_addr = record->ldst_addr + st_diff;
        }
        else{
            current_ldst_addr = 0;
        }
        record->func((uint32_t *)&record->args_list, current_ldst_addr);
    }
}

static int32_t vloop_gen_diff(uint32_t step_num, 
                                int32_t loop2_step, int32_t loop1_step, int32_t loop0_step, 
                                uint32_t loop2_cnt, uint32_t loop1_cnt, uint32_t loop0_cnt){
    int32_t loop2_addr; 
    int32_t loop1_addr;
    int32_t loop0_addr;
    uint32_t step_cnt;

    // printf("%d %d %d %d %d %d %d\n\r", step_num, loop2_step, loop1_step, loop0_step, loop2_cnt, loop1_cnt, loop0_cnt);
    loop2_addr = 0;
    step_cnt = 0;
    for(int i2=0; i2<=loop2_cnt; i2++){
        loop1_addr = loop2_addr;
        for(int i1=0; i1<=loop1_cnt; i1++){
            loop0_addr = loop1_addr;
            for(int i0=0; i0<=loop0_cnt; i0++){
                if(step_cnt == step_num)
                    return loop0_addr;
                loop0_addr += loop0_step;
                step_cnt++;
            }
            loop1_addr += loop1_step;
        }
        loop2_addr += loop2_step;
    }
    return 0;
}

static void vloop_exec(CPURISCVState *env){
    int32_t ld_diff=0, st_diff=0;
    vloop_state_t *vstate = &env->vloop_state;
    for(int i = 0; i < vstate->loop_num; i++){
        ld_diff = vloop_gen_diff(i, env->vld_loop2_step, env->vld_loop1_step, env->vld_loop0_step,
                                    env->vld_loop2_num, env->vld_loop1_num, env->vld_loop0_num);
        st_diff = vloop_gen_diff(i, env->vst_loop2_step, env->vst_loop1_step, env->vst_loop0_step,
                                    env->vst_loop2_num, env->vst_loop1_num, env->vst_loop0_num);
        vloop_exec_once(vstate, ld_diff, st_diff);
    }
}

void false_vld(uint32_t addr, uint32_t x, uint32_t y, uint32_t z){
    npu_log("VLD ADDR = %d, FLag = %d\n\r", addr, x+y+z);
}

void false_vld_wrapper(uint32_t *args, uint32_t addr){
    false_vld(addr, args[0], args[1], args[2]);
}

void false_vst(uint32_t addr, uint32_t x, uint32_t y, uint32_t z){
    npu_log("VST ADDR = %d, FLag = %d\n\r", addr, x+y+z);
}

void false_vst_wrapper(uint32_t *args, uint32_t addr){
    false_vst(addr, args[0], args[1], args[2]);
}

void helper_false_vld(CPURISCVState *env)
{
    int addr = 100;
    vloop_args_t *args;
    if(env->vloop_state.enabled){
        args = vloop_push_record(&env->vloop_state, VLOOP_LD, false_vld_wrapper, addr);
        vloop_push_args(args, 1);
        vloop_push_args(args, 2);
        vloop_push_args(args, 3);
    }
    else{
        false_vld(addr, 1, 2, 3);
    }
}

void helper_false_vst(CPURISCVState *env)
{
    int addr = 1000;
    vloop_args_t *args;
    if(env->vloop_state.enabled){
        args = vloop_push_record(&env->vloop_state, VLOOP_ST, false_vst_wrapper, addr);
        vloop_push_args(args, 111);
        vloop_push_args(args, 222);
        vloop_push_args(args, 333);
    }
    else{
        false_vst(addr, 111, 222, 333);
    }
}

void helper_false_vloop_start(CPURISCVState *env)
{
    npu_log("vloop start\n\r");
    vloop_setup(&env->vloop_state, 36);
}

void helper_false_vloop_end(CPURISCVState *env)
{
    // void (*funcc)(uint32_t *x, uint32_t y);
    int x[3] = {11, 22, 33};
    npu_log("vloop run (repeat = %d, record = %d)\n\r", env->vloop_state.loop_num, env->vloop_state.loop_tail_ptr);
    vloop_exec(env);
    // env->vloop_state.records[0].func(&x[0], 1234);
    vloop_finish(&env->vloop_state);
    npu_log("vloop end\n\r");
}

void helper_necho(CPURISCVState *env)
{
    // int tmp;
    printf("echo at pc=%x\n\r", env->pc);
    npu_log("echo at pc=%x\n\r", env->pc);
    npu_log("VPRO=%d\n\r", env->nvmr[0].flags[0]);
    // npu_log("check vmr: VMRO=%d\n\r", env->nvmr[0].flags[0]);
    // cpu_stl_data(env, 0X1000000, 0xdeadbeef);
    // tmp = cpu_ldl_data(env, 0x1000000);
    // npu_log("check shared memory: [0]=0x%x\n\r", tmp);
    // cpu_stl_data(env, 0X1000000, 0);
    // npu_raise_exception();


    //for test
    int mem_begin_addr = 0x90000000, mem_size_addr = 0x90000004;
    long int mem_begin, mem_size;
    FILE *fp_mem, *fp_reg, *fp_csr;
    mem_begin = cpu_ldl_data(env, mem_begin_addr);
    mem_size = cpu_ldl_data(env, mem_size_addr);
    printf("begin: %x, end: %x\n\r", mem_begin, mem_size);
    //print mem
    printf("mem:\n\r");
    int data;
    fp_mem = fopen("mem_dump.txt","w");
    for(int i=0; i<mem_size/64; i++){ //for vsb
        for(int j=63; j>=0; j--){
            data = cpu_ldsb_data(env, mem_begin+i*64+j);
            fprintf(fp_mem, "%02hhx",data);
        }
        fprintf(fp_mem, "\n");
    }
    // for(int i = 0; i < mem_size/64; i++)
    // {
    //     for(int j = 15; j >= 0; j--)
    //     {
    //         // printf("data_addr:%x\n\r", mem_begin_addr + 16*i + j);
    //         data = cpu_ldl_data(env, mem_begin_addr + 16*i + j);
    //         fprintf(fp_mem, "%08x", data);
    //     }
    //     fprintf(fp_mem, "%c", '\n');
    // }
    fclose(fp_mem);
    // for(int i = mem_begin; i < mem_begin + mem_size; i++){
    //     data = cpu_ldl_data(env, i);
    //     printf("%x ", data);

    // }
    printf("\n============\n\r");

    //print csr
    printf("reg list:\n\r");
    fp_reg = fopen("reg_dump.txt","w");
    for(int i = 0; i < 32; i++)
    {
        fprintf(fp_reg, "%08x\n", env->gpr[i]);
        printf("reg[%d]: %x\n", i, env->gpr[i]);
    }
    close(fp_reg);
    printf("============\n\r");

    //print regs
    printf("csr list:\n\r");
    fp_csr = fopen("csr_dump.txt","w");
    fprintf(fp_csr, "mhartid: %08x\n", env->mhartid);
    fprintf(fp_csr, "mstatus: %08x\n", env->mstatus);
    fprintf(fp_csr, "mtvec: %08x\n", env->mtvec);
    fprintf(fp_csr, "mcause: %08x\n", env->mcause);
    close(fp_csr);
    printf("mhartid: %x\n\r", env->mhartid);
    printf("mstatus: %x\n\r", env->mstatus);
    printf("mtvec: %x\n\r", env->mtvec);
    printf("mcause: %x\n\r", env->mcause);
    printf("============\n\r");

}

#endif
