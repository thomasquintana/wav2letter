-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

-- Install the restserver
-- luarocks install restserver-xavante

require 'io'
require 'torch'
require 'nn'
require 'wav2letter'

local restserver = require("restserver")
local sndfile = require 'sndfile'
local tnt = require 'torchnet'
local xlua = require 'xlua'
local serial = require 'wav2letter.runtime.serial'
local data = require 'wav2letter.runtime.data'
local transforms = require 'wav2letter.runtime.transforms'
local utils = require 'wav2letter.utils'
local decoder = require 'wav2letter.runtime.decoder'

torch.setdefaulttensortype('torch.FloatTensor')

local function cmdtestoptions(cmd)
    cmd:text()
    cmd:text('SpeechRec (c) Ronan Collobert 2015')
    cmd:text()
    cmd:text('Arguments:')
    cmd:argument('-model', 'the trained model!')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-gpu', 1, 'gpu device')
    cmd:option('-nolsm', false, 'remove lsm layer')
    cmd:option('-addlsm', false, 'add lsm layer')
    cmd:option('-dict', "", 'letters.lst')
    cmd:option('-letters', "", 'letter-rep.lst')
    cmd:option('-words', "", 'words.lst')
    cmd:option('-maxword', -1, 'maximum number of words to use')
    cmd:option('-lm', "", 'lm.arpa.bin')
    cmd:option('-smearing', "none", 'none, max or logadd')
    cmd:option('-lmweight', 1, 'lm weight')
    cmd:option('-wordscore', 0, 'word insertion weight')
    cmd:option('-silweight', 0, 'weight for sil')
    cmd:option('-unkscore', -math.huge, 'unknown (word) insertion weight')
    cmd:option('-beamsize', 25000, 'max beam size')
    cmd:option('-beamscore', 40, 'beam score threshold')
    cmd:option('-forceendsil', false, 'force end sil')
    cmd:option('-logadd', false, 'use logadd instead of max')
    cmd:option('-nthread', 0, 'number of threads to use')
    cmd:option('-port', 8080, 'The port number for the rest endpoint to listen on')
    cmd:text()
 end
 
 if #arg < 1 then
    error(string.format([[
 usage:
    %s -model <options...>
 ]], arg[0]))
 end
 
 local reload = arg[1]
 local opt = serial.parsecmdline{
    closure = cmdtestoptions,
    arg = arg,
    default = serial.loadmodel(reload).config.opt
 }
 
 if opt.gpu > 0 then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.gpu)
    cutorch.manualSeedAll(opt.seed)
 end
 
 --dictionaries
 local dict = data.newdict{
    path = opt.dict
 }
 
 local dict39phn
 if opt.target == "phn" then
    dict39phn = data.dictcollapsephones{dictionary=dict}
    if opt.dict39 then
       dict = dict39phn
    end
 end
 
 if opt.dictsil then
    data.dictadd{dictionary=dict, token='N', idx=assert(dict['|'])}
    data.dictadd{dictionary=dict, token='L', idx=assert(dict['|'])}
 end
 
 if opt.ctc or opt.garbage then
    data.dictadd{dictionary=dict, token="#"} -- blank
 end
 
 if opt.replabel > 0 then
    for i=1,opt.replabel do
       data.dictadd{dictionary=dict, token=string.format("%d", i)}
    end
 end
 
 print(string.format('| number of classes (network) = %d', #dict))
 
 --reloading network
 print(string.format('| reloading model <%s>', reload))
 local model = serial.loadmodel{filename=reload, arch=true}
 local network = model.arch.network
 local transitions = model.arch.transitions
 local config = model.config
 local kw = model.config.kw
 local dw = model.config.dw
 assert(kw and dw, 'kw and dw could not be found in model archive')
 
 if opt.nolsm then
    for i=network:size(),1,-1 do
       if torch.typename(network:get(i)) == 'nn.LogSoftMax' then
          print('! removing nn.LogSoftMax layer ' .. i)
          network:remove(i)
       end
    end
 end
 assert(not (opt.addlsm and opt.nolsm))
 if opt.addlsm then
    if opt.gpu then
       network:insert(nn.LogSoftMax():cuda(), network:size())
    else
       network:add(nn.LogSoftMax())
    end
 end
 print(network)
 
 -- make sure we do not apply aug on this
 opt.aug = false
 opt.shift = opt.shift or 0
 
 local criterion
 
 if opt.msc then
    criterion = nn.MultiStateFullConnectCriterion(#dict/opt.nstate, opt.nstate)
 else
    criterion = (opt.ctc and nn.ConnectionistTemporalCriterion(#dict, nil)) or nn.Viterbi(#dict)
 end
 if not opt.ctc then
    criterion.transitions:copy(transitions)
 end
 
 if opt.shift > 0 then
    print(string.format("| using shift scheme (shift=%d dshift=%d)", opt.shift, opt.dshift))
    network = nn.ShiftNet(network, opt.shift)
 end
 
 local function tostring(tensor)
     local str = {}
     tensor:apply(
        function(idx)
           local letter = dict[idx]
           assert(letter)
           table.insert(str, letter)
        end
     )
     return table.concat(str)
  end

local decoder = decoder(
   opt.letters,
   opt.words,
   opt.lm,
   opt.smearing,
   opt.maxword
)

local dopt = {
   lmweight = opt.lmweight,
   wordscore = opt.wordscore,
   unkscore = opt.unkscore,
   beamsize = opt.beamsize,
   beamscore = opt.beamscore,
   forceendsil = opt.forceendsil,
   logadd = opt.logadd,
   silweight = opt.silweight
}

local transformations = transforms.inputfromoptions(opt, kw, dw)

-- Start the web service.
local server = restserver:new():port(opt.port)

server:add_resource("asr", {
   {
      method = "POST",
      path = "/",
      consumes = "application/json",
      produces = "application/json",
      handler = function(request)
         local fwav = sndfile.SndFile(request.file)
         local fwavinfo = fwav:info()
         local wav = fwav:readFloat(fwavinfo.frames)
         fwav:close()

         local netoutput = network:forward(transformations(wav))
         local predictions = criterion:viterbi(netoutput)
         raw_predictions = utils.uniq(predictions)

         if (request.raw_only == true) then
          result = string.format('{"raw_output": %s', tostring(raw_predictions))
          return restserver.response():status(200):entity(result)
         else
          local predictions, lpredictions = decoder(dopt, transitions, network.output)
          predictions = decoder.removeunk(predictions)
          predictions = decoder.tensor2string(predictions)
   
          result = string.format('{"output": %s, "raw_output": %s',
                                 predictions, tostring(raw_predictions))
   
          return restserver.response():status(200):entity(result)
         end
      end,
   },
})

-- This loads the restserver.xavante plugin
server:enable("restserver.xavante"):start()
