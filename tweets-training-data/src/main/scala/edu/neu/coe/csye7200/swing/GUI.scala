package edu.neu.coe.csye7200.swing

import scala.swing._
import scala.swing.event._

object GUI extends SimpleSwingApplication {
//  def top: Frame = new MainFrame {
//    title = "My Frame"
//    contents = new GridPanel(2, 2) {
//      hGap = 3
//      vGap = 3
//      contents += new Button {
//        text = "Press Me!"
//        reactions += {
//          case ButtonClicked(_) => text = "Hello Scala"
//        }
//      }
//    }
//    size = new Dimension(300, 80)
//  }
  var msg = "original message"
    def top = new MainFrame{
      title = "tweets-traning-data"
      var textField = new TextField()
      var button = new Button { text = "run" }
      var label = new Label { text = "there is no result"}
      contents = new BoxPanel(Orientation.Vertical) {
        contents += textField
        contents += button
        contents += label
        border = Swing.EmptyBorder(30, 30, 10, 30)
      }
      size = new Dimension(400,300)
      var clicks = 0
      listenTo(button)
      reactions += {
        case ButtonClicked(b) if b == button =>
          //下面是点击button之后要做的操作
          //获取文本信息用textField.text
          //结果展示放到lavel.text = 这里
          clicks += 1
          label.text = "Number of button clicks:" + clicks + textField.text
      }
    }
}
