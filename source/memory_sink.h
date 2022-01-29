#include "MicroBit.h"
#include "CodalConfig.h"
#include "DataStream.h"

#pragma once

class MemorySink : public DataSink
{
    DataSource      &upstream;
    uint8_t *output;
    int length;
    volatile bool done;
    int currentPosition;

    public:
    MemorySink(DataSource &source, uint8_t *output, int length): 
        upstream(source), output(output), length(length), done(false), currentPosition(0) {
        source.connect(*this);
    }
    virtual ~MemorySink() {}

    virtual int pullRequest() {
        if (done) {
            return DEVICE_BUSY;
        }

        ManagedBuffer b = upstream.pull();
        int size = min(length - currentPosition, b.length());
        memcpy(&output[currentPosition], &b[0], size);

        currentPosition += size;
        if (currentPosition == length) {
            done = true;
        }
        return DEVICE_OK;
    }
    void reset() {
        currentPosition = 0;
        done = false;
    }
    bool isDone() {
        return done;
    }
};
