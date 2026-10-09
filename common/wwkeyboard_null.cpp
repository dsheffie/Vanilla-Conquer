#include "wwkeyboard.h"

/*
** Keyboard backend for builds with no system input, such as the remaster dll on
** platforms other than Windows. Input arrives through the dll interface instead.
*/
class WWKeyboardClassNull : public WWKeyboardClass
{
public:
    virtual void Fill_Buffer_From_System(void)
    {
    }

    virtual KeyASCIIType To_ASCII(unsigned short key)
    {
        return KA_NONE;
    }
};

WWKeyboardClass* CreateWWKeyboardClass(void)
{
    return new WWKeyboardClassNull;
}
