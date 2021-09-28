#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys, getopt
import re
import subprocess

QEMU_PATH = "/home/jwhuang/Work/qemu-install/qemu6-install/bin/qemu-system-riscv32"

def acquire_mem_range(config_file):
   begin_mem = ''
   size_mem = ''
   data_addr = ''
   file = open(config_file)
   context = file.readlines()
   file.close()
   for line in context:
      begin = re.findall(r'dump_addr=(0x[0-9, A-F]*)', line, re.M|re.I)
      # end = re.findall(r'^end:(0x[0-9, A-F]*)', line, re.M|re.I)
      size = re.findall(r'dump_size=(0x[0-9, A-F]*)', line, re.M|re.I)
      data_a = re.findall(r'data_addr=(0x[0-9, A-F]*)', line, re.M|re.I)
      if begin:
         begin_mem = begin[0]
      if size:
         size_mem = size[0]
      if data_a:
         data_addr = data_a[0]
   print('begin:', begin_mem)
   print('size:', size_mem)
   print('data_addr', data_addr)

   return begin_mem, size_mem, data_addr


def main(argv):
   elf_file = ''
   config_file = ''
   data_file = ''
   command = ''
   begin = 0
   size = 0
   data_addr = 0
   try:
      opts, args = getopt.getopt(argv,":",["elf=","config=","data="])
   except getopt.GetoptError:
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt in ("--elf"):
         elf_file = arg
      elif opt in ("--config"):
         config_file = arg
      elif opt in ("--data"):
         data_file = arg
   print('elf file is:', elf_file)
   print('config file is', config_file)
   print('data file is', data_file)
   
   begin, size, data_addr = acquire_mem_range(config_file)
   # begin = str(begin)
   # begin = int(begin, 16)
   # end = str(end)
   # end = int(end, 16)
   print(type(begin))
   command = QEMU_PATH + ' -machine sifive_e -nographic -kernel ' + elf_file \
            + ' -device loader,addr=0x90000000,data=' + str(begin) + ',data-len=4 ' \
            + '-device loader,addr=0x90000004,data=' + str(size) + ',data-len=4 ' \
            #+ '-device loader,file=' + data_file + ',addr=' + str(data_addr)
   print(command)
   p=subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
   for line in p.stdout.readlines():
        print(line)
   

if __name__ == "__main__":
   main(sys.argv[1:])